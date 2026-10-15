from pricing import black76_call, black76_put

F = 1.638
r = 0.0049
T = 369 / 365
sigma = 0.40

cap_value = black76_call(
    F=F,
    K=1.68,
    r=r,
    T=T,
    sigma=sigma
)

floor_value = black76_put(
    F=F,
    K=1.40,
    r=r,
    T=T,
    sigma=sigma
)

print(f"Cap benchmark: ${cap_value:.4f} per gallon")
print(f"Floor benchmark: ${floor_value:.4f} per gallon")

# Basic financial sanity checks
assert cap_value >= 0
assert floor_value >= 0

print("Pricing functions passed basic checks.")