# CHILI: Chemically-Informed Large-scale Inorganic Nanomaterials Dataset for Advancing Graph Machine Learning
Ulrik Friis-Jensen $^{1,2,†}$, Frederik L. Johansen $^{1,2,†}$, Andy S. Anker $^{3,4}$, Erik B. Dam $^{2}$, Kirsten M. Ø. Jensen $^{1}$, Raghavendra Selvan $^{2}$

1: Department of Chemistry, University of Copenhagen, Denmark

2: Department of Computer Science, University of Copenhagen, Denmark

3: Department of Energy Conversion and Storage, Technical University of Denmark, Denmark

4: University of Oxford, United Kingdom

† Both authors contributed equally to this research.

### Abstract
Advances in graph machine learning (ML) have been driven by applications in chemistry as graphs have remained the most expressive representations of molecules. 
This has led to progress within both fields, as challenging chemical data has helped improve existing methods and to develop new ones.
While early graph ML methods focused primarily on small organic molecules, more recently, the scope of graph ML has expanded to include inorganic materials. Modelling the periodicity and symmetry of inorganic crystalline materials poses unique challenges, which existing graph ML methods are unable to immediately address. The focus on inorganic nanomaterials further increases complexity as the scale of number of nodes within each graph can be broad ($10$ to $10^5$).

In addition, the bulk of existing graph ML focuses on characterising molecules by predicting target properties with graphs as input. The most exciting applications of graph ML will be in their generative capabilities, in order to explore the vast chemical space from a data-driven perspective. Currently, generative modelling of graphs is not at par with other domains such as images or text, as generating chemically valid molecules and materials of varying properties is not straightforward. 

In this work, we invite the graph ML community to address these open challenges by presenting two new chemically-informed large-scale inorganic (`CHILI`) nanomaterials datasets.  These datasets contain nanomaterials of different scales and properties represented as graphs of varying sizes. The first dataset is a medium-scale dataset (with overall >6M nodes, >49M edges) of mono-metallic oxide nanomaterials generated from $12$ selected crystal types (`CHILI-3K`). This dataset has a narrower chemical scope focused on an interesting part of chemical space with a lot of active research. The second is a large-scale dataset (with overall >183M nodes, >1.2B edges) of nanomaterials generated from experimentally determined crystal structures (`CHILI-100K`). The crystal structures used in `CHILI-100K` are obtained from a curated subset from the Crystallography Open Database (COD) and has a broader chemical scope covering database entries for 68 metals and 11 non-metals. We define 11 property prediction tasks covering node-, edge-, and graph- level tasks that span classification and regression. In addition we also define structure prediction tasks, which are of special interest for nanomaterial research. We benchmark the performance of a wide array of baseline methods starting with simple baselines to multiple off-the-shelf graph neural networks. 
Based on these benchmarking results, we highlight areas which need future work to achieve useful performance for applications in (nano)materials chemistry. To the best of our knowledge, `CHILI-3K` and `CHILI-100K` are the first open-source nanomaterial datasets of this scale -- both on the individual graph level and of the dataset as a whole -- and the only nanomaterials datasets with high structural and elemental diversity.
