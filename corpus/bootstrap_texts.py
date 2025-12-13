"""
DAIMON Corpus Bootstrap.

Foundational texts for the wisdom corpus.

Usage:
    from corpus.bootstrap_texts import bootstrap_corpus
    bootstrap_corpus()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .manager import CorpusManager, TextMetadata

logger = logging.getLogger("daimon.corpus.bootstrap")

# Foundational texts
BOOTSTRAP_TEXTS = [
    # Stoic Philosophy
    {
        "author": "Marcus Aurelius",
        "title": "Meditations - On Morning Preparation",
        "category": "filosofia/estoicos",
        "content": """Begin the morning by saying to thyself, I shall meet with the busy-body,
the ungrateful, arrogant, deceitful, envious, unsocial. All these things happen to them
by reason of their ignorance of what is good and evil. But I who have seen the nature
of the good that it is beautiful, and of the bad that it is ugly, and the nature of him
who does wrong, that it is akin to me; I can neither be injured by any of them, for no
one can fix on me what is ugly, nor can I be angry with my kinsman, nor hate him.""",
        "themes": ["virtue", "stoicism", "preparation", "equanimity"],
        "source": "Meditations, Book II, 1",
        "relevance_score": 0.9,
    },
    {
        "author": "Epictetus",
        "title": "Enchiridion - On Control",
        "category": "filosofia/estoicos",
        "content": """Some things are in our control and others not. Things in our control are
opinion, pursuit, desire, aversion, and, in a word, whatever are our own actions.
Things not in our control are body, property, reputation, command, and, in one word,
whatever are not our own actions. The things in our control are by nature free,
unrestrained, unhindered; but those not in our control are weak, slavish, restrained,
belonging to others.""",
        "themes": ["control", "stoicism", "dichotomy", "freedom"],
        "source": "Enchiridion, Chapter 1",
        "relevance_score": 0.95,
    },
    {
        "author": "Seneca",
        "title": "Letters - On the Shortness of Life",
        "category": "filosofia/estoicos",
        "content": """It is not that we have a short time to live, but that we waste a lot of it.
Life is long enough, and a sufficiently generous amount has been given to us for the
highest achievements if it were all well invested. But when it is wasted in heedless
luxury and spent on no good activity, we are forced at last by death's final constraint
to realize that it has passed away before we knew it was passing.""",
        "themes": ["time", "stoicism", "mortality", "wisdom"],
        "source": "De Brevitate Vitae, 1",
        "relevance_score": 0.85,
    },

    # Greek Philosophy
    {
        "author": "Aristotle",
        "title": "Nicomachean Ethics - On the Good",
        "category": "filosofia/gregos",
        "content": """Every art and every inquiry, and similarly every action and pursuit,
is thought to aim at some good; and for this reason the good has rightly been declared
to be that at which all things aim. But a certain difference is found among ends;
some are activities, others are products apart from the activities that produce them.""",
        "themes": ["ethics", "teleology", "virtue", "purpose"],
        "source": "Nicomachean Ethics, Book I, Chapter 1",
        "relevance_score": 0.9,
    },
    {
        "author": "Plato",
        "title": "Republic - Allegory of the Cave",
        "category": "filosofia/gregos",
        "content": """Allegory of the Cave: Imagine prisoners chained in a cave, only seeing shadows
on the wall cast by objects passing in front of a fire behind them. This is their reality.
If one prisoner is freed and sees the fire, then the outside world, and finally the sun,
he realizes the shadows were not reality. Returning to tell the others, they would not
believe him, preferring their familiar shadows. The philosopher's task is to lead others
from darkness to light, from opinion to knowledge.""",
        "themes": ["knowledge", "truth", "enlightenment", "education"],
        "source": "Republic, Book VII, 514a-520a",
        "relevance_score": 0.9,
    },
    {
        "author": "Socrates",
        "title": "Apology - On Wisdom",
        "category": "filosofia/gregos",
        "content": """I am wiser than this man; for neither of us really knows anything fine
and good, but this man thinks he knows something when he does not, whereas I, as I do
not know anything, do not think I do either. I seem, then, in just this little thing
to be wiser than this man at any rate, that what I do not know I do not think I know either.""",
        "themes": ["wisdom", "humility", "knowledge", "self-knowledge"],
        "source": "Apology, 21d",
        "relevance_score": 0.9,
    },

    # Logic and Reasoning
    {
        "author": "Aristotle",
        "title": "Organon - On Valid Reasoning",
        "category": "logica",
        "content": """A syllogism is discourse in which, certain things being stated, something
other than what is stated follows of necessity from their being so. I mean by the last
phrase that it follows because of them, and by this, that no further term is required
from without in order to make the consequence necessary. A perfect syllogism is one
which needs nothing other than what has been stated to make plain what necessarily follows.""",
        "themes": ["logic", "reasoning", "syllogism", "deduction"],
        "source": "Prior Analytics, Book I, Chapter 1",
        "relevance_score": 0.85,
    },

    # Ethics
    {
        "author": "Immanuel Kant",
        "title": "Groundwork - Categorical Imperative",
        "category": "etica",
        "content": """Act only according to that maxim whereby you can at the same time will
that it should become a universal law. There is therefore but one categorical imperative,
namely, this: Act only on that maxim whereby thou canst at the same time will that it
should become a universal law. Now if all imperatives of duty can be deduced from this
one imperative as from their principle, then we shall be able to show what we understand
by duty.""",
        "themes": ["ethics", "duty", "universality", "morality"],
        "source": "Groundwork of the Metaphysics of Morals, Section II",
        "relevance_score": 0.85,
    },

    # Science and Method
    {
        "author": "Karl Popper",
        "title": "Logic of Scientific Discovery - Falsifiability",
        "category": "ciencia",
        "content": """The criterion of the scientific status of a theory is its falsifiability,
or refutability, or testability. It is easy to obtain confirmations, or verifications,
for nearly every theory if we look for confirmations. Confirmations should count only
if they are the result of risky predictions; that is to say, if, unenlightened by the
theory in question, we should have expected an event which was incompatible with the
theory, an event which would have refuted the theory.""",
        "themes": ["science", "method", "falsifiability", "epistemology"],
        "source": "Conjectures and Refutations, Chapter 1",
        "relevance_score": 0.85,
    },
    {
        "author": "Richard Feynman",
        "title": "On Scientific Method",
        "category": "ciencia",
        "content": """The first principle is that you must not fool yourself, and you are the
easiest person to fool. So you have to be very careful about that. After you've not
fooled yourself, it's easy not to fool other scientists. You just have to be honest
in a conventional way after that. I'm talking about a specific, extra type of integrity
that is not lying, but bending over backwards to show how you're maybe wrong.""",
        "themes": ["science", "honesty", "integrity", "method"],
        "source": "Cargo Cult Science, 1974",
        "relevance_score": 0.9,
    },
]


def bootstrap_corpus(corpus_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Bootstrap the wisdom corpus with foundational texts.

    Args:
        corpus_path: Optional custom corpus path

    Returns:
        Dict with bootstrap results
    """
    manager = CorpusManager(corpus_path)
    added = 0
    skipped = 0

    for text_data in BOOTSTRAP_TEXTS:
        author = str(text_data["author"])
        title = str(text_data["title"])
        text_id = manager.generate_text_id(author, title)

        if text_id in manager.texts:
            logger.debug("Skipping existing text: %s", text_id)
            skipped += 1
            continue

        themes_raw = text_data.get("themes")
        themes: List[str] = []
        if themes_raw is not None and isinstance(themes_raw, list):
            themes = [str(t) for t in themes_raw]

        relevance_raw = text_data.get("relevance_score", 0.5)
        relevance = float(relevance_raw) if isinstance(relevance_raw, (int, float)) else 0.5

        metadata = TextMetadata(
            source=str(text_data.get("source", "")),
            relevance_score=relevance,
            themes=themes,
        )

        manager.add_text(
            author=author,
            title=title,
            category=str(text_data["category"]),
            content=str(text_data["content"]),
            metadata=metadata,
        )
        added += 1

    logger.info("Bootstrap complete: %d added, %d skipped", added, skipped)

    return {
        "added": added,
        "skipped": skipped,
        "total": len(BOOTSTRAP_TEXTS),
        "corpus_stats": manager.get_stats(),
    }


if __name__ == "__main__":
    print("DAIMON Corpus Bootstrap")
    print("=" * 50)

    result = bootstrap_corpus()

    print("\nResults:")
    print(f"  Added: {result['added']}")
    print(f"  Skipped: {result['skipped']}")
    print(f"  Total available: {result['total']}")

    print("\nCorpus stats:")
    stats = result["corpus_stats"]
    print(f"  Total texts: {stats['total_texts']}")
    print(f"  Authors: {stats['total_authors']}")
    print(f"  Themes: {stats['total_themes']}")
    print(f"  Path: {stats['corpus_path']}")
