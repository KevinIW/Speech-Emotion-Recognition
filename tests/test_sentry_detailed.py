"""
Advanced Sentry testing script with more detailed event tracking.
This script will help identify if events are being sent correctly and how to find them.
"""
import sentry_sdk
from dotenv import load_dotenv
import os
import time
import uuid
import sys
from sentry_sdk import capture_message, capture_exception, set_tag, set_user

def print_section(title):
    """Print a section header for better readability."""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50)

# Load environment variables
print_section("LOADING ENVIRONMENT")
load_dotenv(override=True)

sentry_dsn = os.environ.get("SENTRY_DSN")
environment = os.environ.get("ENVIRONMENT", "test")

if not sentry_dsn:
    print("❌ SENTRY_DSN not found in environment variables")
    sys.exit(1)

print(f"✅ Using DSN: {sentry_dsn[:15]}...{sentry_dsn[-5:]}")
print(f"✅ Environment: {environment}")

# Generate unique identifiers for this test run
test_id = str(uuid.uuid4())[:8]
print(f"✅ Test ID: {test_id} (use this to search in Sentry)")

# Initialize Sentry with more detailed configuration
print_section("INITIALIZING SENTRY")
try:
    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=environment,
        release=f"ser-test-{test_id}",  # Create a unique release for this test
        traces_sample_rate=1.0,         # Capture all transactions
        send_default_pii=True,          # Send personal identifiable information
        debug=True                      # Enable debug mode for verbose output
    )
    print("✅ Sentry initialized successfully with debug=True")
except Exception as e:
    print(f"❌ Failed to initialize Sentry: {e}")
    sys.exit(1)

# Set tags and context to make events easier to find
print_section("SETTING CONTEXT")
set_tag("test_run", test_id)
set_tag("test_type", "detailed_diagnostics")
set_user({"id": f"test-user-{test_id}", "email": "test@example.com"})

sentry_sdk.set_context("test_info", {
    "script": "test_sentry_detailed.py",
    "timestamp": time.time(),
    "test_id": test_id
})
print("✅ Added tags, user, and context to help identify events")

# Test message capture
print_section("SENDING TEST MESSAGE")
event_id = capture_message(
    f"Test message from SER app (ID: {test_id})",
    level="info"
)
print(f"✅ Test message sent with event_id: {event_id}")

# Test error capture
print_section("SENDING TEST ERROR")
try:
    # Create a more structured error with stack trace
    def inner_function():
        # This will create a proper stack trace
        raise ValueError(f"Test exception for Sentry (ID: {test_id})")
    
    inner_function()
except Exception as e:
    event_id = capture_exception(e)
    print(f"✅ Test exception captured with event_id: {event_id}")

# Test performance monitoring
print_section("TESTING PERFORMANCE MONITORING")
with sentry_sdk.start_transaction(op="test", name=f"Test Transaction {test_id}"):
    # Create a span to time a specific operation
    with sentry_sdk.start_span(op="sleep", description="Simulated work"):
        time.sleep(0.5)  # Simulate work
    
    # Create another span
    with sentry_sdk.start_span(op="calculation", description="Complex calculation"):
        # Simulate complex work
        result = sum(i * i for i in range(100000))
        print(f"✅ Calculation result: {result}")

print("✅ Performance transaction completed")

# Flush events and provide guidance
print_section("FINALIZING")
print("Waiting for events to be sent to Sentry...")
sentry_sdk.flush(timeout=5)
print("✅ Sentry events flushed")

print_section("HOW TO FIND YOUR EVENTS")
print("1. Go to your Sentry dashboard: https://sentry.io")
print("2. Navigate to Issues, Performance, or Releases section")
print(f"3. Search for the test ID: {test_id}")
print(f"4. Look for the release name: ser-test-{test_id}")
print("5. Filter by environment: test")
print("\nIf you still can't see events, check these locations:")
print("- Issues > All: Check for your exception")
print("- Performance > Transactions: Look for your test transaction")
print("- Releases > ser-test-{}: Check for your specific release")
print("- Settings > Projects > [Your Project] > Client Keys")
print("  Ensure your DSN is correct and enabled")

print_section("TROUBLESHOOTING")
print("If no events appear after several minutes:")
print("1. Verify your network connection")
print("2. Check if your firewall is blocking Sentry connections")
print("3. Verify your DSN is correct in .env")
print("4. Ensure your project is properly set up in Sentry dashboard")
print("5. Try adding SENTRY_DEBUG=1 to your environment variables")
