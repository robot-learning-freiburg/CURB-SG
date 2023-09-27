# CURB-SG: Collaborative Dynamic 3D Scene Graphs for Automated Driving
[**arXiv**](https://arxiv.org/abs/2309.06635) | [**Website**](http://curb.cs.uni-freiburg.de/) | [**Video**](https://www.youtube.com/watch?v=qbzQNz7_i8c)

This repository is the official implementation of the paper:

> **Collaborative Dynamic 3D Scene Graphs for Automated Driving**
>
> [Elias Greve]()&ast;, [Martin Büchner](https://rl.uni-freiburg.de/people/buechner)&ast;, [Niclas Vödisch](https://vniclas.github.io/)&ast;, [Wolfram Burgard](https://www.utn.de/person/wolfram-burgard/), and [Abhinav Valada](https://rl.uni-freiburg.de/people/valada). <br>
> &ast;Equal contribution. <br> 
> 
> *arXiv preprint arXiv:2309.06635*, 2023

<p align="center">
  <img src="./assets/curb_overview.png" alt="Overview of SPINO approach" width="800" />
</p>

If you find our work useful, please consider citing our paper:
```
@article{greve2023curb,
  title={Collaborative Dynamic 3D Scene Graphs for Automated Driving},
  author={Greve, Elias and Büchner, Martin and Vödisch, Niclas and Burgard, Wolfram and Valada, Abhinav},
  journal={arXiv preprint arXiv:2309.06635},
  year={2023}
}
```


## 📔 Abstract

Maps have played an indispensable role in enabling safe and automated driving. Although there have been many advances on different fronts ranging from SLAM to semantics, building an actionable hierarchical semantic representation of urban dynamic scenes from multiple agents is still a challenging problem. In this work, we present Collaborative URBan Scene Graphs (CURB-SG) that enable higher-order reasoning and efficient querying for many functions of automated driving. CURB-SG leverages panoptic LiDAR data from multiple agents to build large-scale maps using an effective graph-based collaborative SLAM approach that detects inter-agent loop closures. To semantically decompose the obtained 3D map, we build a lane graph from the paths of ego agents and their panoptic observations of other vehicles. Based on the connectivity of the lane graph, we segregate the environment into intersecting and non-intersecting road areas. Subsequently, we construct a multi-layered scene graph that includes lane information, the position of static landmarks and their assignment to certain map sections, other vehicles observed by the ego agents, and the pose graph from SLAM including 3D panoptic point clouds. We extensively evaluate CURB-SG in urban scenarios using a photorealistic simulator.


## 👩‍💻 Code

We will release the code upon the acceptance of our paper.


## 👩‍⚖️  License

For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
For any commercial purpose, please contact the authors.


## 🙏 Acknowledgment

This work was funded by the European Union’s Horizon 2020 research and innovation program grant No 871449-OpenDR and the German Research Foundation (DFG) Emmy Noether Program grant No 468878300.
<br><br>
<p float="left">
  <a href="https://opendr.eu/"><img src="./assets/opendr_logo.png" alt="drawing" height="100"/></a>
  &nbsp;
  &nbsp;
  &nbsp;
  <a href="https://www.dfg.de/en/research_funding/programmes/individual/emmy_noether/index.html"><img src="./assets/dfg_logo.png" alt="drawing" height="100"/></a>  
</p>

