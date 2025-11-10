# pachamama

Algorithmic trading on [Alpaca](https://alpaca.markets/), name is from [Wikipedia article on Alpacas](https://en.wikipedia.org/wiki/Alpaca):
> Alpacas are closely tied to cultural practices for Andeans people. Prior to colonization, the image of the alpaca was used in rituals and in their religious practices. Since the people in the region depended heavily on these animals for their sustenance, the alpaca was seen as a gift from Pachamama.

```mermaid
---
config:
  theme: redux-dark
  layout: dagre
---
flowchart TD
    n1["Data Ingestion &amp;<br>Aggregation"] --> n2["Database"]
    n2 --> n3["Algorithm"]
    n3 --> n5["Signal Aggregation &amp;<br>Weighting"]
    n5 -- Order Executions --> n4["API Clients"]
    n4 -- Market<br>Data --> n1
    n2@{ shape: db}
    n3@{ shape: procs}
    n4@{ shape: procs}
```

