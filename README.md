# CHILI: Chemically-Informed Large-scale Inorganic Nanomaterials Dataset for Advancing Graph Machine Learning
Ulrik Friis-Jensen $^{1,2,†}$, Frederik L. Johansen $^{1,2,†}$, Andy S. Anker $^3$, Erik B. Dam $^{2}$, Kirsten M. Ø. Jensen $^{1}$, Raghavendra Selvan $^{2}$

1: Department of Chemistry, University of Copenhagen

2: Department of Computer Science, University of Copenhagen

3: Department of Energy Conversion and Storage, Technical University of Denmark

† Both authors contributed equally to this research.

### Abstract
Advances in graph machine learning (ML) have been driven by applications in chemistry as graphs have remained the most expressive representations of molecules. 
This has led to progress within both fields, as challenging chemical data has enabled the improvement of existing methods and the development of new ones. 
While early graph ML methods focused primarily on organic- or small inorganic- molecules, more recently, the scope of graph ML has expanded to include crystalline materials. Modelling crystalline materials poses challenges that are unique, and existing graph ML methods are not immediately capable of addressing them. For instance, when crystalline materials are represented as graphs the scale of number of nodes within each graph can be broad: $\mathcal{O}(10)-\mathcal{O}(10^5)$. The periodicity and symmetry of crystalline materials pose additional challenges to existing graph ML methods. 

In addition, the bulk of existing graph ML focuses on characterising molecules by predicting target properties with graphs as input. The most exciting applications of graph ML will be in their generative capabilities, in order to explore the vast chemical space from a data-driven perspective. Currently, generative modelling of graphs is not at par with other domains such as images or text, as generating chemically valid molecules of varying properties is not straightforward. 

In this work, we invite the graph ML community to address these open challenges by presenting two new chemically-informed large-scale inorganic (`CHILI`) nanomaterials datasets.  These datasets contain nanomaterials of different scales and properties represented as graphs of varying sizes. The first dataset is a medium-scale dataset (with overall >6M nodes, >49M edges) of mono-metallic oxide nanomaterials generated from theoretically ideal crystal structures (`CHILI-3K`). This dataset has a narrower chemical scope focused on an interesting part of chemical space with a lot of active research. The second is a large-scale dataset (with overall >183M nodes, >1.2B edges) of nanomaterials generated from experimentally determined crystal structures (`CHILI-100K`). The crystal structures used in `CHILI-100K` are a subset from the Crystallography Open Database (COD) and has a broader chemical scope covering database entries for 68 metals and 11 non-metals. We define 11 prediction tasks covering node-, edge-, and graph-level tasks as well as both classification and regression. In addition we also define generative tasks, which are of special interest for nanomaterial research. We benchmark the performance of a wide-array of baseline methods starting with simple baselines to multiple off-the-shelf graph neural networks appropriate for the proposed tasks on both datasets. Based on these benchmarking results, we highlight areas which need future work to achieve useful performance for applications in (nano)materials chemistry. To the best of our knowledge, `CHILI-3K` and `CHILI-100K` are the first open-source nanomaterial datasets of this scale -- both on the individual graph level and of the dataset as a whole -- and the only nanomaterials dataset with high structural and elemental diversity.
