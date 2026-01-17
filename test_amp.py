from amplitude import Amplitude, BaseEvent
from datetime import datetime

# Your API key
AMPLITUDE_API_KEY = "c850ba20162c9c24e43f2b3bc6602aa5"  # Replace with your actual key

# Initialize
client = Amplitude(AMPLITUDE_API_KEY)

# Send test event
print("Sending test event to Amplitude...")

event = BaseEvent(
    event_type="Test Event",
    user_id="test_user_123",
    event_properties={
        'test_property': 'hello world',
        'timestamp': datetime.now().isoformat()
    }
)

client.track(event)
client.flush()

print("✓ Test event sent!")
print("\nNow check Amplitude:")
print("1. Go to amplitude.com")
print("2. Click 'Users' in left sidebar")
print("3. Click 'User Look-Up'")
print("4. Search for:   test_user_123")
print("5. You should see 'Test Event'")