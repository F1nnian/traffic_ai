# Very basic first example

- Two streets with cars only going in one direction each
- No acceleration, same velocity for every car
- Changing phases should require some time (yellow-phase) so phases dont get switched too often

## States

| Feature         | Description                                | Categories                               |
| --------------- | ------------------------------------------ | ---------------------------------------- |
| `current_phase` | Which street has green light               | 0 = NS (North-South), 1 = EW (East-West) |
| `sq_wait_NS`    | Sum of squared waiting times for NS-street | 0: = 0, 1: 1-100, 2: 101–400, 3: >400    |
| `sq_wait_EW`    | Sum of squared waiting times for EW-street | Same categories                          |

## Actions

0: Stay in current phase
1: Switch Phase

## Reward Function

The reward function guides the traffic light AI to make decisions that:

- Minimize overall waiting times
- Avoid “starvation” (cars waiting excessively long)

`reward = - sum(wait_time**2 for all cars)`

- squaring ensures that longer waiting times are penalized more

Maybe add a term for the car with the longest waiting time, so very long waiting times are reduced further.
