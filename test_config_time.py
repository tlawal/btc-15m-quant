
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo
from config import Config

def test_preferred_trading_hours(monkeypatch):
    """
    Test the is_preferred_trading_time logic with various mocked times.
    Preferred: 7:00 AM - 4:10:30 PM ET on Weekdays.
    Note: 4:30 PM is 16.5.
    """
    
    def set_mock_time(year, month, day, hour, minute, weekday_idx=None):
        # Mon=0, Sun=6. Let's pick a date that matches the weekday_idx.
        # 2026-03-23 is a Monday (idx=0).
        # We can just mock datetime.now directly.
        class MockDateTime:
            @classmethod
            def now(cls, tz=None):
                dt = datetime(year, month, day, hour, minute, tzinfo=tz)
                return dt
        monkeypatch.setattr("config.datetime", MockDateTime)

    # 1. Inside: Monday 9:00 AM ET -> True
    set_mock_time(2026, 3, 23, 9, 0) 
    assert Config.is_preferred_trading_time() is True

    # 2. Border Start: Monday 7:00 AM ET -> True
    set_mock_time(2026, 3, 23, 7, 0)
    assert Config.is_preferred_trading_time() is True

    # 3. Just Before Start: Monday 6:59 AM ET -> False
    set_mock_time(2026, 3, 23, 6, 59)
    assert Config.is_preferred_trading_time() is False

    # 4. Border End: Monday 4:30 PM ET -> False (strict < end)
    set_mock_time(2026, 3, 23, 16, 30)
    assert Config.is_preferred_trading_time() is False

    # 5. Just Before End: Monday 4:29 PM ET -> True
    set_mock_time(2026, 3, 23, 16, 29)
    assert Config.is_preferred_trading_time() is True

    # 6. Well After End: Monday 5:00 PM ET -> False
    set_mock_time(2026, 3, 23, 17, 0)
    assert Config.is_preferred_trading_time() is False

    # 7. Weekend Check: Saturday 12:00 PM ET -> False
    set_mock_time(2026, 3, 21, 12, 0) # Saturday
    assert Config.is_preferred_trading_time() is False

    # 8. Sunday Check: Sunday 12:00 PM ET -> False
    set_mock_time(2026, 3, 22, 12, 0) # Sunday
    assert Config.is_preferred_trading_time() is False

if __name__ == "__main__":
    pytest.main([__file__])
