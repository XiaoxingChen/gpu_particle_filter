<body>
<h1>Particle Filter Localization with GPU</h1>
<p>
When considering using GPU to parallelize programs on laptop or mobile phone, CUDA may not be available on these devices. OpenCL allows one to call GPU functions even on ARM platform. This demo shows how to use OpenCL to accelerate a Particle Filter algorithm for robot localization.
</p>
<h1>Examples</h1>
<h2>Ice World</h2>
<div style="vertical-align:middle; text-align:center">
        <img src="https://user-images.githubusercontent.com/16934019/80937927-a1e1c880-8e09-11ea-99cb-6fa34e88f895.gif" alt="em" align="center" width="70%">
</div>

<h2>Featureless Corridor</h2>
<div style="vertical-align:middle; text-align:center">
        <img src="https://user-images.githubusercontent.com/16934019/80937931-aa3a0380-8e09-11ea-9937-49a308e7e95d.gif" alt="em" align="center" width="70%">
</div>

<h2>Circular Corridor</h2>
<p>This seems an impossible task for a localization solution without observable global pose.</p>
<div style="vertical-align:middle; text-align:center">
        <img src="https://user-images.githubusercontent.com/16934019/80937934-ac03c700-8e09-11ea-95ee-2300c933b5f3.gif" alt="em" align="center" width="70%">
</div>

<h1>How to Run</h1>
<h2>Setup</h2>

```
git clone https://github.com/XiaoxingChen/gpu_particle_filter
cd gpu_particle_filter
pip3 install -r requirements.txt
source setup.bash
```
<h3>Intel GPU + Ubuntu 18.04</h3>

```
add-apt-repository ppa:intel-opencl/intel-opencl
apt-get update
apt-get install intel-opencl-icd
```

```
sudo apt install ocl-icd-opencl-dev
```
<h3>NVDIA GPU + Ubuntu 18.04</h3>

```
sudo apt install ocl-icd-opencl-dev
```
<h3>Windows Subsystem for Linux</h3>
<p>
Currently not available for OpenCL. Please give up.
</p>

<h3>ARM Mali GPU</h3>
<p>Should be available after I convert the python part to C++ version.</p>

<h2>Run Demo</h2>

```
python3 demo/localization.py ice_world
```
```
python3 demo/localization.py corridor
```
```
python3 demo/localization.py circle_corridor
```
<p><a href="https://steamcommunity.com/sharedfiles/filedetails/?id=258081127">Why is ice world?</a></p>
<h1>Tested Platforms</h1>
<ul>
<li>Intel UHD Graphics 620 (rev 07), Ubuntu 18.04 [pass]</li>
<li>Intel HD Graphics 5500 (rev 09), Ubuntu 18.04 [freeze sometimes]</li>
<li>NVIDIA GF119 [NVS 315] (rev a1), Ubuntu 18.04 [pass]</li>
</ul>


<h1>References</h1>
<p>
    <ol class="bib">
        <li>
        Thrun, Sebastian. 
        <cite>
            <a href="http://www.probabilistic-robotics.org/">"Probabilistic Robotics."</a>
        </cite>
            Communications of the ACM 45.3 (2002): 52-57.
        </li>
        <li>
        Terejanu, Gabriel A. 
        <cite>
            <a href="https://cse.sc.edu/~terejanu/files/tutorialMC.pdf">"Tutorial on Monte Carlo Techniques." </a>
        </cite>
            Department of Computer Science and Engineering. University at Buffalo (2009).
        </li>
        <li>
            Atsushi, Sakai. 
            <a href="https://github.com/AtsushiSakai/PythonRobotics/blob/master/Localization/particle_filter/particle_filter.py">"PythonRobotics."</a>
        </li>
        <li>
            Xiaoxing, Chen. 
            <a href="https://github.com/XiaoxingChen/particle_filter_localization_demo">"Particle Filter Localization Demo."</a>
        </li>
    </ol>
</p>
<h2>Author</h2>
<ul>
<li>Xiaoxing Chen</li>
</ul>
</body>