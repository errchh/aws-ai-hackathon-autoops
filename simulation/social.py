"""
Social Media Sentiment Simulator for Healthcare and Wellness Trends

This module simulates social media sentiment and trending topics
related to healthcare and wellness products and lifestyles.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import random

logger = logging.getLogger(__name__)


class SocialSentimentSimulator:
    """
    Simulates social media sentiment and trending topics for healthcare
    and wellness products, including influencer content and consumer trends.
    """

    def __init__(self):
        self.current_sentiment: Dict[
            str, float
        ] = {}  # category -> sentiment score (-1.0 to 1.0)
        self.trending_topics: Dict[str, Dict[str, Any]] = {}
        self.influencer_content: List[Dict[str, Any]] = []
        self.sentiment_history: List[Dict[str, Any]] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize social sentiment simulation"""
        if self._initialized:
            return

        logger.info("Initializing social media sentiment simulator...")

        # Initialize baseline sentiment for different categories
        self.current_sentiment = {
            "immune_support": 0.3,  # Generally positive
            "fitness_gear": 0.4,  # Very positive
            "stress_relief": 0.2,  # Moderately positive
            "digestive_health": 0.1,  # Neutral to positive
            "essential_oils": 0.5,  # Highly positive
            "vitamins": 0.25,  # Positive
            "meditation": 0.35,  # Positive
            "organic_products": 0.3,  # Positive
        }

        # Initialize trending topics
        self.trending_topics = {
            "immune_support_trending": {
                "topic": "winter immunity boost",
                "sentiment": 0.6,
                "mentions_24h": 2500,
                "influencer_reach": 50000,
                "peak_time": "winter",
                "related_products": ["vit_c_500mg", "vit_d_1000iu", "probiotics_50b"],
            },
            "fitness_new_year": {
                "topic": "New Year fitness resolutions",
                "sentiment": 0.7,
                "mentions_24h": 1800,
                "influencer_reach": 75000,
                "peak_time": "january",
                "related_products": ["yoga_mat", "resistance_bands"],
            },
            "stress_relief_trending": {
                "topic": "work-life balance tips",
                "sentiment": 0.4,
                "mentions_24h": 3200,
                "influencer_reach": 45000,
                "peak_time": "year_round",
                "related_products": ["lavender_oil", "meditation_cushion"],
            },
        }

        self._initialized = True
        logger.info("Social sentiment simulator initialized")

    async def update_sentiment(self, current_time: datetime) -> None:
        """Update social sentiment based on current time and trends"""
        if not self._initialized:
            return

        # Update seasonal sentiment patterns
        current_month = current_time.month

        # Winter months boost immune support sentiment
        if current_month in [12, 1, 2, 3]:
            self.current_sentiment["immune_support"] = min(
                0.8, self.current_sentiment["immune_support"] + 0.1
            )

        # January boosts fitness sentiment
        if current_month == 1:
            self.current_sentiment["fitness_gear"] = min(
                0.9, self.current_sentiment["fitness_gear"] + 0.15
            )

        # May (Mental Health Month) boosts stress relief
        if current_month == 5:
            self.current_sentiment["stress_relief"] = min(
                0.7, self.current_sentiment["stress_relief"] + 0.2
            )

        # Add random sentiment fluctuations (±0.1)
        for category in self.current_sentiment:
            fluctuation = random.uniform(-0.1, 0.1)
            self.current_sentiment[category] = max(
                -1.0, min(1.0, self.current_sentiment[category] + fluctuation)
            )

        # Update trending topics
        await self._update_trending_topics(current_time)

        # Generate influencer content occasionally
        if random.random() < 0.3:  # 30% chance per update
            await self._generate_influencer_content(current_time)

        # Store sentiment history
        self.sentiment_history.append(
            {
                "timestamp": current_time.isoformat(),
                "sentiment": self.current_sentiment.copy(),
                "active_trends": len(
                    [
                        t
                        for t in self.trending_topics.values()
                        if t["mentions_24h"] > 1000
                    ]
                ),
            }
        )

        # Keep only last 100 entries
        if len(self.sentiment_history) > 100:
            self.sentiment_history = self.sentiment_history[-100:]

    async def _update_trending_topics(self, current_time: datetime) -> None:
        """Update trending topics based on time and random events"""
        current_month = current_time.month

        # Seasonal topic updates
        if current_month == 1 and "fitness_new_year" in self.trending_topics:
            # Boost New Year fitness topic
            self.trending_topics["fitness_new_year"]["mentions_24h"] = min(
                5000,
                self.trending_topics["fitness_new_year"]["mentions_24h"]
                + random.randint(200, 500),
            )

        if (
            current_month in [12, 1, 2]
            and "immune_support_trending" in self.trending_topics
        ):
            # Boost immune support in winter
            self.trending_topics["immune_support_trending"]["mentions_24h"] = min(
                4000,
                self.trending_topics["immune_support_trending"]["mentions_24h"]
                + random.randint(150, 400),
            )

        # Random topic emergence (small chance)
        if random.random() < 0.1:
            await self._create_new_trend(current_time)

        # Decay old trends
        for topic_id, topic in self.trending_topics.items():
            if topic["mentions_24h"] > 500:
                decay = random.randint(50, 150)
                topic["mentions_24h"] = max(0, topic["mentions_24h"] - decay)

    async def _create_new_trend(self, current_time: datetime) -> None:
        """Create a new trending topic"""
        trend_templates = [
            {
                "topic": "morning wellness routine",
                "sentiment": 0.5,
                "base_mentions": 1200,
                "related_products": ["vit_c_500mg", "lavender_oil"],
            },
            {
                "topic": "natural sleep remedies",
                "sentiment": 0.4,
                "base_mentions": 900,
                "related_products": ["meditation_cushion", "lavender_oil"],
            },
            {
                "topic": "gut health awareness",
                "sentiment": 0.3,
                "base_mentions": 1100,
                "related_products": ["probiotics_50b"],
            },
            {
                "topic": "essential oil diffuser recipes",
                "sentiment": 0.6,
                "base_mentions": 1400,
                "related_products": ["lavender_oil", "eucalyptus_oil"],
            },
        ]

        template = random.choice(trend_templates)
        trend_id = f"trend_{int(current_time.timestamp())}"

        self.trending_topics[trend_id] = {
            "topic": template["topic"],
            "sentiment": template["sentiment"],
            "mentions_24h": template["base_mentions"],
            "influencer_reach": random.randint(20000, 60000),
            "peak_time": "current",
            "related_products": template["related_products"],
            "created_at": current_time.isoformat(),
        }

        logger.info(f"New social trend emerged: {template['topic']}")

    async def _generate_influencer_content(self, current_time: datetime) -> None:
        """Generate influencer content that affects sentiment"""
        influencer_types = [
            {
                "type": "fitness_trainer",
                "reach": 25000,
                "categories": ["fitness_gear", "vitamins"],
            },
            {
                "type": "wellness_coach",
                "reach": 18000,
                "categories": ["stress_relief", "meditation"],
            },
            {
                "type": "nutrition_expert",
                "reach": 22000,
                "categories": ["vitamins", "organic_products"],
            },
            {
                "type": "holistic_healer",
                "reach": 15000,
                "categories": ["essential_oils", "stress_relief"],
            },
        ]

        influencer = random.choice(influencer_types)
        category = random.choice(influencer["categories"])

        # Generate content
        content = {
            "timestamp": current_time.isoformat(),
            "influencer_type": influencer["type"],
            "reach": influencer["reach"],
            "category": category,
            "sentiment_impact": random.uniform(0.1, 0.3),
            "content_type": random.choice(["post", "story", "reel"]),
            "engagement_rate": random.uniform(0.02, 0.08),
        }

        self.influencer_content.append(content)

        # Apply sentiment impact
        if category in self.current_sentiment:
            old_sentiment = self.current_sentiment[category]
            self.current_sentiment[category] = min(
                1.0, old_sentiment + content["sentiment_impact"]
            )

            logger.info(
                f"Influencer content boosted {category} sentiment: "
                f"{old_sentiment:.2f} → {self.current_sentiment[category]:.2f}"
            )

        # Keep only recent content
        if len(self.influencer_content) > 50:
            self.influencer_content = self.influencer_content[-50:]

    async def get_current_sentiment(self) -> Dict[str, Any]:
        """Get current social sentiment data"""
        return {
            "category_sentiment": self.current_sentiment.copy(),
            "active_trends": len(
                [t for t in self.trending_topics.values() if t["mentions_24h"] > 1000]
            ),
            "total_influencer_content": len(self.influencer_content),
            "top_trends": sorted(
                self.trending_topics.items(),
                key=lambda x: x[1]["mentions_24h"],
                reverse=True,
            )[:3],
        }

    async def get_trending_topics(
        self, min_mentions: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get currently trending topics above threshold"""
        return [
            {
                "topic": topic["topic"],
                "sentiment": topic["sentiment"],
                "mentions_24h": topic["mentions_24h"],
                "influencer_reach": topic["influencer_reach"],
                "related_products": topic["related_products"],
            }
            for topic in self.trending_topics.values()
            if topic["mentions_24h"] >= min_mentions
        ]

    async def get_sentiment_for_category(self, category: str) -> Optional[float]:
        """Get sentiment score for a specific category"""
        return self.current_sentiment.get(category)

    async def trigger_viral_content(self, category: str, impact: float = 0.4) -> None:
        """Trigger viral content that significantly boosts sentiment"""
        if category in self.current_sentiment:
            old_sentiment = self.current_sentiment[category]
            self.current_sentiment[category] = min(1.0, old_sentiment + impact)

            # Create viral trend
            viral_trend = {
                "topic": f"viral_{category}_trend_{int(datetime.now().timestamp())}",
                "sentiment": min(0.9, old_sentiment + impact),
                "mentions_24h": random.randint(5000, 15000),
                "influencer_reach": random.randint(100000, 500000),
                "peak_time": "viral",
                "related_products": [],  # Would be populated based on category
            }

            trend_id = f"viral_{int(datetime.now().timestamp())}"
            self.trending_topics[trend_id] = viral_trend

            logger.info(
                f"Viral content triggered for {category}: sentiment "
                f"{old_sentiment:.2f} → {self.current_sentiment[category]:.2f}"
            )

    async def get_sentiment_history(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get sentiment history for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        return [
            entry
            for entry in self.sentiment_history
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff_time
        ]

    async def reset(self) -> None:
        """Reset social sentiment to initial state"""
        await self.initialize()
        self.sentiment_history = []
        self.influencer_content = []

        logger.info("Social sentiment simulator reset to initial state")
