import numpy as np
import pandas as pd

RNG = np.random.default_rng(2026)
N = 600

brands = ["Samsung", "Xiaomi", "Apple", "OnePlus", "Realme", "Vivo"]
brand_weights = [0.25, 0.22, 0.18, 0.15, 0.10, 0.10]
brand = RNG.choice(brands, size=N, p=brand_weights)

user_intensity = RNG.normal(0, 1, N)

screen_on_time = np.clip(RNG.normal(5.5, 1.5, N) + 1.2 * user_intensity, 1.5, 12.0)
gaming_hours = np.clip(RNG.poisson(np.clip(1.2 + 0.8 * user_intensity, 0.1, None), N), 0, 7)
brightness_pct = np.clip(RNG.normal(60, 18, N) + 8.0 * user_intensity, 20, 100)
bg_apps = RNG.poisson(np.clip(4 + 1.5 * user_intensity, 1, None))

max_temp_c = np.clip(32.0 + 1.2 * gaming_hours + 0.08 * brightness_pct + 0.3 * bg_apps + RNG.normal(0, 1.5, N), 30.0, 52.0)

battery_drain_rate = (6.0 
                      + 1.8 * gaming_hours 
                      + 0.07 * brightness_pct 
                      + 0.4 * bg_apps 
                      + 0.35 * (max_temp_c - 35.0) 
                      + RNG.normal(0, 1.2, N))
battery_drain_rate = np.round(np.clip(battery_drain_rate, 4.0, 30.0), 1)

battery_health = np.clip(100 - (RNG.normal(12, 4, N) + 1.5 * gaming_hours + 0.4 * max_temp_c), 65, 100)

s_health = battery_health
s_drain = np.clip((25.0 - battery_drain_rate) / 20.0, 0, 1) * 100
s_temp = np.clip((50.0 - max_temp_c) / 20.0, 0, 1) * 100
s_sot = np.clip(screen_on_time / 10.0, 0, 1) * 100

performance_index = (0.35 * s_health + 0.30 * s_drain + 0.20 * s_temp + 0.15 * s_sot)
performance_index = np.round(np.clip(performance_index, 0, 100), 1)

logit = -5.0 + 0.18 * max_temp_c + 0.4 * gaming_hours - 0.04 * performance_index
throttling = (RNG.random(N) < 1 / (1 + np.exp(-logit))).astype(int)

df = pd.DataFrame({
    "device_id": [f"DEV{1000+i}" for i in range(N)],
    "brand": brand,
    "screen_on_time_h": np.round(screen_on_time, 1),
    "gaming_hours_h": gaming_hours,
    "brightness_pct": np.round(brightness_pct, 1),
    "bg_apps_count": bg_apps,
    "max_temp_c": np.round(max_temp_c, 1),
    "battery_drain_rate_pct_h": battery_drain_rate,
    "battery_health_pct": np.round(battery_health, 1),
    "performance_index": performance_index,
    "throttled": throttling
})

df.to_csv("data/students.csv", index=False)
print(f"Generated {len(df)} records -> data/students.csv")