# Tree-structured Attention with Hierarchical Accumulation for Query Plan Cardinality and Cost Prediction

## Ackonwledge

For the realization of the paper Tree-structured Attention with Hierarchical Accumulation [Tree-structured Attention with Hierarchical Accumulation](https://arxiv.org/abs/2002.08046).

We use it in the database to perdict the cost of a plan and cardinality of the query plan.


## Environment

## Experiment

| version | parpmeter                                          | result                                                                                                                                                      |
| ------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| v1.0    | d_model=128,<br> d_ff=128, N=2, lr=0.001, epoch=10 | max qerror: 662924.5300 <br> mean qerror: 1018.3936 <br> media qerror: 3.1462<br> 90th qerror: 23.3711 <br> 95th qerror: 51.8297 <br> 99th qerror: 756.4599 |
| v1.1    | d_model=521,dff=128, N=2, lr=0.001, epoch=10       |max qerror: 892079.6152 <br> mean qerror: 2151.0649 <br> media qerror: 3.1404 <br> 90th qerror: 31.9187 <br> 95th qerror: 72.9243 <br> 99th qerror: 2229.1361|