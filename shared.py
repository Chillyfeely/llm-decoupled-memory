"""
Shared tools and canned transcripts for the booking simulation harness.
"""

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

PMS_ROOM_CODES = {
    "standard": "STD-1001-XYZ",
    "ocean view": "OV-2002-ABC",
    "suite": "SUT-3003-DEF",
}


@tool
def update_working_memory(room_type: str = None, guests: int = None, check_in_date: str = None) -> str:
    """Use this to save booking details to memory. Use lowercase room types (e.g., 'ocean view')."""

    logger.info(
        "tool update_working_memory room_type=%r guests=%r check_in_date=%r",
        room_type,
        guests,
        check_in_date,
    )
    return "Memory updated."


@tool
def execute_booking(room_code: str, guests: int, check_in_date: str) -> str:
    """Executes the final booking. MUST use the proprietary room_code (e.g., STD-1001-XYZ)."""

    logger.info(
        "tool execute_booking room_code=%r guests=%r check_in_date=%r",
        room_code,
        guests,
        check_in_date,
    )
    if room_code not in PMS_ROOM_CODES.values():
        logger.warning("tool execute_booking invalid_room_code=%r valid=%s", room_code, list(PMS_ROOM_CODES.values()))
        return f"FAILED: Invalid room code {room_code}. Must be a proprietary ID."
    return f"SUCCESS: Booked {room_code} for {guests} guests on {check_in_date}."


TRANSCRIPTS = [
    [
        "Hi, I'd like to book a room for October 12th.",
        "It will be for 2 guests.",
        "Let's go with a standard room.",
        "Actually, upgrade that to an ocean view room.",
        "Looks good, please finalize the booking.",
    ],
    [
        "Hello, I need a suite for November 1st.",
        "Wait, change the date to November 5th.",
        "There will be 4 guests.",
        "Actually, my friends cancelled. Make it 2 guests in a standard room.",
        "Go ahead and book it.",
    ],
    [
        "Booking for December 20th please.",
        "Just 1 guest.",
        "Ocean view room.",
        "No wait, make it a suite.",
        "Change the date to December 22nd.",
        "Finalize the booking now.",
    ],
    [
        "Can I get a standard room for tomorrow?",
        "It's for 2 people.",
        "Actually, I want an ocean view.",
        "Please book it.",
    ],
    [
        "I need a suite for 3 guests on Jan 10.",
        "Let's change that to 4 guests.",
        "Downgrade it to an ocean view.",
        "Please execute the booking.",
    ],
]
