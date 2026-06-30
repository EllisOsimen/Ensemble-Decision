How I plan the agreement to work

| Situation                             | Example             | Output                               | Confidence |
| ------------------------------------- | ------------------- | ------------------------------------ | ---------- |
| 3 agree                               | `[1,1,1]`           | `1`                                  | High       |
| 2 agree on organ                      | `[1,1,0]`           | `1`                                  | Medium     |
| 2 agree on background                 | `[0,0,1]`           | `0`, unless connected to organ       | Medium-low |
| 2 agree on organ, 1 says other organ  | `[3,3,2]`           | `3`                                  | Medium     |
| all disagree, 1 background + 2 organs | `[0,1,3]`           | weighted/local decision              | Low        |
| all disagree, 3 organs                | `[1,2,3]`           | weighted/local decision or uncertain | Very low   |
| all disagree and no strong evidence   | `[0,1,2]` ambiguous | mark uncertain                       | Very low   |
