"""
Assigns survey questions to respondents under dataset, mechanism,
and redundancy constraints. Least-answered-first with soft reservations
for concurrency.
"""

import random
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Set

from pymongo import UpdateOne


# Survey design constants
NUM_QUESTIONS_PER_SURVEY = 15
REDUNDANCY_FACTOR = 4  # Each variant must be answered 4 times

# Coverage targets
EXPECTED_DATASETS = 10
EXPECTED_METHODS = 3  # DPBart, DPMLM, DPPrompt
EXPECTED_EPSILONS = 3  # 750, 1500, 2500
EXPECTED_PAIRINGS = EXPECTED_METHODS * EXPECTED_EPSILONS  # 9


def _effective_count(question):
    """answered_count + assigned_count (uses precomputed field if available)."""
    if "effective_count" in question:
        return question["effective_count"]
    return question.get("answered_count", 0) + question.get("assigned_count", 0)


def _priority_key(question):
    """Lower count first, then light randomization to avoid deterministic ties."""
    return (_effective_count(question), question.get("answered_count", 0), random.random())


def get_eligible_variants(questions_collection):
    """Returns questions with effective_count < REDUNDANCY_FACTOR.
    Falls back to all questions if none are under the target."""

    pipeline = [
        {
            "$addFields": {
                "assigned_count": {"$ifNull": ["$assigned_count", 0]},
                "effective_count": {
                    "$add": ["$answered_count", {"$ifNull": ["$assigned_count", 0]}]
                },
            }
        },
        {"$match": {"effective_count": {"$lt": REDUNDANCY_FACTOR}}},
    ]

    eligible = list(questions_collection.aggregate(pipeline))

    if not eligible:
        print("⚠️  All questions have reached redundancy factor, allowing all questions")
        eligible = list(questions_collection.find({}))
        for q in eligible:
            q["effective_count"] = q.get("answered_count", 0) + q.get(
                "assigned_count", 0
            )

    return eligible


def select_one_per_dataset(eligible_variants, questions_collection):
    """Picks the least-answered question from each dataset."""


    by_dataset = defaultdict(list)
    for question in eligible_variants:
        by_dataset[question["dataset"]].append(question)

    selected = []
    selected_variant_ids = set()


    for dataset, questions in by_dataset.items():

        questions_sorted = sorted(
            questions,
            key=lambda x: (
                x.get("effective_count", x.get("answered_count", 0)),
                random.random(),
            ),
        )

        if questions_sorted:
            chosen = questions_sorted[0]
            selected.append(chosen)
            selected_variant_ids.add(chosen["variant_id"])


    remaining = [
        q for q in eligible_variants if q["variant_id"] not in selected_variant_ids
    ]

    print(f"  ✓ Selected 1 question per dataset: {len(selected)} questions")

    return selected, remaining


def ensure_mechanism_coverage(selected_variants, remaining_variants, max_questions=None):
    """Adds questions until all method x epsilon pairings are covered."""


    covered_pairings = set()
    for q in selected_variants:
        pairing = (q["method"], q["epsilon"])
        covered_pairings.add(pairing)

    missing_pairings = set()

    all_pairings = set()
    for q in selected_variants + remaining_variants:
        all_pairings.add((q["method"], q["epsilon"]))

    missing_pairings = all_pairings - covered_pairings

    if not missing_pairings:
        print(
            f"  ✓ Mechanism coverage already satisfied: {len(covered_pairings)}/9 pairings"
        )
        return selected_variants, remaining_variants

    print(f"  ⚠️  Need {len(missing_pairings)} more pairings: {missing_pairings}")


    selected_variant_ids = set(q["variant_id"] for q in selected_variants)

    for missing_pairing in missing_pairings:
        if max_questions is not None and len(selected_variants) >= max_questions:
            break

        method, epsilon = missing_pairing


        candidates = [
            q
            for q in remaining_variants
            if q["method"] == method
            and q["epsilon"] == epsilon
            and q["variant_id"] not in selected_variant_ids
        ]

        if candidates:
            candidates_sorted = sorted(candidates, key=_priority_key)
            chosen = candidates_sorted[0]

            selected_variants.append(chosen)
            selected_variant_ids.add(chosen["variant_id"])
            remaining_variants.remove(chosen)

            print(f"    ✓ Added {method} × {epsilon}")

    final_pairings = set((q["method"], q["epsilon"]) for q in selected_variants)
    print(f"  ✓ Mechanism coverage satisfied: {len(final_pairings)} pairings")

    return selected_variants, remaining_variants


def fill_remaining_slots(selected_variants, remaining_variants, num_questions):
    """Tops up to num_questions with the least-answered remaining variants."""
    slots_needed = num_questions - len(selected_variants)

    if slots_needed <= 0:
        print(f"  ✓ Already have {len(selected_variants)} questions")
        return selected_variants

    print(f"Filling {slots_needed} remaining slots...")


    remaining_sorted = sorted(
        remaining_variants,
        key=_priority_key,
    )


    to_add = remaining_sorted[:slots_needed]
    selected_variants.extend(to_add)

    print(
        f"  ✓ Added {len(to_add)} questions to reach {len(selected_variants)} questions"
    )

    return selected_variants


def plan_assignment_from_variants(variants, num_questions=NUM_QUESTIONS_PER_SURVEY):
    """Builds an assignment from an in-memory variant list, preferring
    under-target variants and preserving coverage goals."""
    if not variants:
        return []


    under_target = [q for q in variants if _effective_count(q) < REDUNDANCY_FACTOR]
    primary_pool = under_target if len(under_target) >= num_questions else variants

    selected, remaining = select_one_per_dataset(primary_pool, None)
    selected, remaining = ensure_mechanism_coverage(
        selected, remaining, max_questions=num_questions
    )
    selected = fill_remaining_slots(selected, remaining, num_questions)

    # Fallback: pull from the full pool if we're still short.
    if len(selected) < num_questions:
        selected_ids = {q["variant_id"] for q in selected}
        fallback = sorted(
            [q for q in variants if q["variant_id"] not in selected_ids],
            key=_priority_key,
        )
        selected.extend(fallback[: num_questions - len(selected)])

    return selected[:num_questions]


def reserve_questions(questions_collection, variant_ids, user_id):
    """Soft-locks variants by incrementing assigned_count and recording the session."""

    timestamp = datetime.now(tz=timezone.utc)


    bulk_operations = [
        UpdateOne(
            {"variant_id": vid},
            {
                "$inc": {"assigned_count": 1},
                "$push": {
                    "assigned_sessions": {"user_id": user_id, "assigned_at": timestamp}
                },
            },
        )
        for vid in variant_ids
    ]

    try:
        result = questions_collection.bulk_write(bulk_operations, ordered=False)
        print(f"  ✓ Reserved {len(variant_ids)} questions for user {user_id}")
        return True
    except Exception as e:
        print(f"  ✗ Error reserving questions: {e}")
        return False


def release_questions(questions_collection, variant_ids, user_id):
    """Removes soft locks on submission or timeout cleanup."""

    # Only release if this user has a reservation, to avoid negative drift.
    bulk_operations = [
        UpdateOne(
            {"variant_id": vid, "assigned_sessions.user_id": user_id},
            {
                "$inc": {"assigned_count": -1},
                "$pull": {"assigned_sessions": {"user_id": user_id}},
            },
        )
        for vid in variant_ids
    ]

    try:
        result = questions_collection.bulk_write(bulk_operations, ordered=False)
        print(f"  ✓ Released {len(variant_ids)} questions for user {user_id}")
        return True
    except Exception as e:
        print(f"  ✗ Error releasing questions: {e}")
        return False


def cleanup_stale_reservations(questions_collection, timeout_hours=2):
    """Removes reservations older than timeout_hours. Call on app startup."""

    cutoff_time = datetime.now(tz=timezone.utc) - timedelta(hours=timeout_hours)

    print(f"\n🧹 Cleaning up reservations older than {timeout_hours} hours...")


    pipeline = [
        {"$match": {"assigned_sessions": {"$exists": True, "$ne": []}}},
        {
            "$project": {
                "variant_id": 1,
                "stale_sessions": {
                    "$filter": {
                        "input": "$assigned_sessions",
                        "as": "session",
                        "cond": {"$lt": ["$$session.assigned_at", cutoff_time]},
                    }
                },
            }
        },
        {"$match": {"stale_sessions": {"$ne": []}}},
    ]

    stale_questions = list(questions_collection.aggregate(pipeline))

    if not stale_questions:
        print("  ✓ No stale reservations found")
        return 0

    print(f"  Found {len(stale_questions)} questions with stale reservations")


    bulk_operations = []
    total_stale_sessions = 0

    for q in stale_questions:
        stale_count = len(q["stale_sessions"])
        total_stale_sessions += stale_count


        bulk_operations.append(
            UpdateOne(
                {"variant_id": q["variant_id"]},
                {
                    "$inc": {"assigned_count": -stale_count},
                    "$pull": {
                        "assigned_sessions": {"assigned_at": {"$lt": cutoff_time}}
                    },
                },
            )
        )


    try:
        result = questions_collection.bulk_write(bulk_operations, ordered=False)
        print(
            f"  ✓ Cleaned up {total_stale_sessions} stale reservations from {result.modified_count} questions"
        )
        return total_stale_sessions
    except Exception as e:
        print(f"  ✗ Error during cleanup: {e}")
        return 0


def assign_survey_questions(
    questions_collection, user_id, num_questions=NUM_QUESTIONS_PER_SURVEY
) -> List[Dict]:
    """Assigns num_questions to a respondent with dataset/mechanism coverage
    and reserves them via soft lock. Returns shuffled question documents."""
    print(f"\nAssigning {num_questions} survey questions for user {user_id}...")

    # Load question pool with live effective counts.
    print("Step 1: Loading question pool...")
    pipeline = [
        {
            "$addFields": {
                "assigned_count": {"$ifNull": ["$assigned_count", 0]},
                "effective_count": {
                    "$add": ["$answered_count", {"$ifNull": ["$assigned_count", 0]}]
                },
            }
        }
    ]
    all_questions = list(questions_collection.aggregate(pipeline))
    print(f"  ✓ Loaded {len(all_questions)} total questions")

    if len(all_questions) == 0:
        raise ValueError("No questions found in database")

    # Plan assignment, prioritizing under-target variants.
    print("Step 2: Planning assignment...")
    selected = plan_assignment_from_variants(all_questions, num_questions=num_questions)

    if len(selected) < num_questions:
        raise ValueError(
            f"Could only select {len(selected)} questions (expected {num_questions})"
        )

    # Reserve.
    print("Step 3: Reserving questions...")
    variant_ids = [q["variant_id"] for q in selected]
    if not reserve_questions(questions_collection, variant_ids, user_id):
        raise ValueError("Failed to reserve questions")

    # Shuffle.
    print("Step 4: Shuffling...")
    random.shuffle(selected)


    print("\nAssignment complete - Validation:")
    datasets_covered = set(q["dataset"] for q in selected)
    pairings_covered = set((q["method"], q["epsilon"]) for q in selected)

    print(f"  • Total questions: {len(selected)}")
    print(f"  • Datasets covered: {len(datasets_covered)}/10")
    print(f"  • Mechanism pairings: {len(pairings_covered)}/9")
    print(f"  • Pairings: {sorted(pairings_covered)}")

    if len(datasets_covered) < 10:
        print(
            f"  ⚠️  Warning: Only {len(datasets_covered)} datasets covered (expected 10)"
        )

    if len(pairings_covered) < 9:
        print(
            f"  ⚠️  Warning: Only {len(pairings_covered)} pairings covered (expected 9)"
        )

    return selected


def get_survey_statistics(questions_collection):
    """Returns answered-count distribution and progress toward redundancy target."""

    pipeline = [
        {"$group": {"_id": "$answered_count", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]

    distribution = list(questions_collection.aggregate(pipeline))

    total_questions = sum(d["count"] for d in distribution)

    stats = {
        "total_questions": total_questions,
        "answered_distribution": {d["_id"]: d["count"] for d in distribution},
        "target_responses_per_question": REDUNDANCY_FACTOR,
        "total_target_responses": total_questions * REDUNDANCY_FACTOR,
    }

    return stats
