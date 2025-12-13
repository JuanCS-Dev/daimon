/**
 * DAIMON Browser Watcher - Popup Script
 */

const DAIMON_ENDPOINT = "http://localhost:8003/api/browser/status";

async function checkConnection() {
  const statusDiv = document.getElementById("status");
  const statusDot = document.getElementById("statusDot");

  try {
    const response = await fetch(DAIMON_ENDPOINT, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (response.ok) {
      statusDiv.className = "status connected";
      statusDiv.textContent = "Connected to DAIMON";
      statusDot.className = "dot green";
    } else {
      throw new Error("Server error");
    }
  } catch (e) {
    statusDiv.className = "status disconnected";
    statusDiv.textContent = "DAIMON not running";
    statusDot.className = "dot red";
  }
}

// Check on popup open
checkConnection();

// Refresh every 5 seconds while popup is open
setInterval(checkConnection, 5000);
