/**
 * DAIMON Browser Watcher - Background Service Worker
 * ===================================================
 *
 * Privacy-preserving browser activity tracking.
 * Sends DOMAIN ONLY to local DAIMON server. Never URLs, content, or titles.
 *
 * Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
 */

const DAIMON_ENDPOINT = "http://localhost:8003/api/browser/event";
const HEARTBEAT_INTERVAL = 30000; // 30 seconds

let currentTabId = null;
let currentUrl = null;
let heartbeatTimer = null;
let isConnected = false;

/**
 * Extract domain from URL (privacy-preserving).
 * @param {string} url - Full URL
 * @returns {string|null} - Domain only
 */
function extractDomain(url) {
  if (!url) return null;

  try {
    // Skip internal browser pages
    if (
      url.startsWith("chrome://") ||
      url.startsWith("chrome-extension://") ||
      url.startsWith("about:") ||
      url.startsWith("moz-extension://") ||
      url.startsWith("edge://")
    ) {
      return null;
    }

    const urlObj = new URL(url);
    let domain = urlObj.hostname.toLowerCase();

    // Remove www prefix
    if (domain.startsWith("www.")) {
      domain = domain.slice(4);
    }

    return domain || null;
  } catch (e) {
    return null;
  }
}

/**
 * Send event to DAIMON server.
 * @param {string} type - Event type
 * @param {object} data - Additional data
 */
async function sendEvent(type, data = {}) {
  const domain = extractDomain(data.url);

  // Don't send if no valid domain
  if (type !== "tab_close" && !domain) {
    return;
  }

  const event = {
    type,
    url: data.url || "",
    domain,
    timestamp: new Date().toISOString(),
  };

  try {
    const response = await fetch(DAIMON_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(event),
    });

    if (response.ok) {
      if (!isConnected) {
        console.log("DAIMON: Connected to server");
        isConnected = true;
      }
    } else {
      console.warn("DAIMON: Server returned", response.status);
      isConnected = false;
    }
  } catch (e) {
    // Server not running - this is expected when DAIMON is off
    if (isConnected) {
      console.log("DAIMON: Server disconnected");
      isConnected = false;
    }
  }
}

/**
 * Handle tab activation (switching tabs).
 */
chrome.tabs.onActivated.addListener(async (activeInfo) => {
  try {
    const tab = await chrome.tabs.get(activeInfo.tabId);
    currentTabId = activeInfo.tabId;
    currentUrl = tab.url;

    await sendEvent("tab_change", { url: tab.url });
  } catch (e) {
    // Tab may have been closed
  }
});

/**
 * Handle tab URL updates (navigation within tab).
 */
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  // Only care about URL changes in active tab
  if (tabId === currentTabId && changeInfo.url) {
    currentUrl = changeInfo.url;
    await sendEvent("tab_change", { url: changeInfo.url });
  }
});

/**
 * Handle tab close.
 */
chrome.tabs.onRemoved.addListener(async (tabId) => {
  if (tabId === currentTabId) {
    currentTabId = null;
    currentUrl = null;
    await sendEvent("tab_close", {});
  }
});

/**
 * Handle window focus changes.
 */
chrome.windows.onFocusChanged.addListener(async (windowId) => {
  if (windowId === chrome.windows.WINDOW_ID_NONE) {
    // Browser lost focus
    await sendEvent("tab_close", {});
  } else {
    // Browser gained focus - get active tab
    try {
      const [tab] = await chrome.tabs.query({
        active: true,
        windowId: windowId,
      });
      if (tab) {
        currentTabId = tab.id;
        currentUrl = tab.url;
        await sendEvent("tab_change", { url: tab.url });
      }
    } catch (e) {
      // Ignore errors
    }
  }
});

/**
 * Periodic heartbeat to confirm current state.
 */
function startHeartbeat() {
  if (heartbeatTimer) {
    clearInterval(heartbeatTimer);
  }

  heartbeatTimer = setInterval(async () => {
    if (currentUrl) {
      await sendEvent("heartbeat", { url: currentUrl });
    }
  }, HEARTBEAT_INTERVAL);
}

/**
 * Initialize on install/startup.
 */
chrome.runtime.onInstalled.addListener(() => {
  console.log("DAIMON Browser Watcher installed");
  startHeartbeat();
});

chrome.runtime.onStartup.addListener(() => {
  console.log("DAIMON Browser Watcher started");
  startHeartbeat();
});

// Start heartbeat immediately
startHeartbeat();

// Get initial state
chrome.tabs.query({ active: true, currentWindow: true }).then(([tab]) => {
  if (tab) {
    currentTabId = tab.id;
    currentUrl = tab.url;
    sendEvent("tab_change", { url: tab.url });
  }
});
