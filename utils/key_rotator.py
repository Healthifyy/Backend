import time

class KeyRotator:
    def __init__(self, keys: list):
        # Filter out empty or whitespace-only keys
        self.keys = [k for k in keys if k and k.strip()]
        self.current_index = 0
        self.rate_limited = {}  # dict of key -> timestamp when rate limited

    def get_key(self) -> str or None:
        if not self.keys:
            return None
            
        num_keys = len(self.keys)
        for _ in range(num_keys):
            key = self.keys[self.current_index]
            
            # Check if key is currently rate limited (cooldown: 60 seconds)
            if key in self.rate_limited:
                if time.time() - self.rate_limited[key] < 60:
                    # Key is still cooling down, move to next
                    self.current_index = (self.current_index + 1) % num_keys
                    continue
                else:
                    # Cooldown period over
                    del self.rate_limited[key]
            
            # Key is available
            self.current_index = (self.current_index + 1) % num_keys
            return key
            
        # All keys are currently rate limited
        return None

    def mark_rate_limited(self, key: str):
        self.rate_limited[key] = time.time()
        print(f"Key rotated due to rate limit. Available keys: {self.available_count}")

    @property
    def available_count(self) -> int:
        now = time.time()
        count = 0
        for key in self.keys:
            if key not in self.rate_limited or (now - self.rate_limited[key] >= 60):
                count += 1
        return count
