# <img src="figs/arena.png" width="24"/> **RobotArena ∞ : Unlimited Robot Benchmarking Via Real-To-Sim Translation**

Welcome to the **RobotArena ∞**, an open-source toolkit for evaluating robotic policies under diverse **scene perturbations**.

Our framework leverages custom-built simulated environments—called **Robot Arenas**—and supports a range of popular models including:

- **CogAct**
- **RoboVLM**
- **Octo**
- **SpatialVLA**
- **open-pi-zero**
- **X-VLA**

---

## 📚 Contents

- [🛠 Environment Setup](#environment-setup)
  - [1. CogAct](#1-environment-setup-for-cogact)
  - [2. RoboVLM](#2-environment-setup-for-robovlm)
  - [3. SpatialVLA & Octo](#3-environment-setup-for-spatialvla-and-octo)
  - [4. open-pi-zero](#4-environment-setup-for-open-pi-zero)
  - [5. X-VLA](#5-environment-setup-for-x-vla)
  - [6. Genesis](#6-environment-setup-for-genesis)
  - [7. Gemini](#7-environment-setup-for-gemini)

- [🚀 Running Evaluation](#running-evaluation)
  - **Policy Servers**
    - [RoboVLM Server](#robovlm-server)
    - [Octo Server](#octo-server)
    - [SpatialVLA Server](#spatialvla-server)
    - [CogAct Server](#cogact-server)
    - [open-pi-zero Server](#open-pi-zero-server)
    - [X-VLA Server](#x-vla-server)
  - [📁 Example Data Structure](#example-data-structure)
  - [📜 Evaluation Scripts](#evaluation-scripts)

- [📦 Output File Structure](#output-file-structure)  
- [🧮 GVL Automated Scoring Script](#gvl-automated-scoring-script)



## 🛠 Environment Setup

* First update the submodules in the repository to ensure all dependencies are correctly initialized:

```bash
git submodule update --init --recursive
```

We have set up separate servers for each policy for evaluation. Please follow the steps below to set up the environment.

### 1. Environment Setup for CogAct

<details>
<summary>environment cogact (click to expand)</summary>

```bash
conda env create -f env/cogact.yml
conda activate cogact
cd ./CogACT
pip install -e .
pip install uvicorn fastapi "tomli>=1.1.0" "rpds-py>=0.7.1" "traitlets>=5.3"
cd ../SimplerEnv
python -m pip install pip==25.0.1
pip install -e .
pip install -r requirements_full_install.txt 
cd ManiSkill2_real2sim
pip install -e .
pip install --upgrade typing_extensions
pip install "numpy<1.25"
pip install tensorflow_datasets==4.9.3
pip install --upgrade pydantic fastapi
cd ../..
cp ./SimplerEnv/simpler_env/policies/sim_cogact/adaptive_ensemble.py ./SimplerEnv/simpler_env/policies/sim_cogact/CogACT/adaptive_ensemble.py
```
</details>

### 2. Environment Setup for RoboVLM


<details>
<summary>environment robovlms (click to expand)</summary>

```bash
conda env create -f env/robovlm.yml
conda activate robovlms
cd ./RoboVLM
pip install -e .
cd ../SimplerEnv
pip install -e .
cd ManiSkill2_real2sim
pip install -e .
pip install "opencv-python<4.10" "numpy<2.0" "pyarrow<21.0.0"
pip install matplotlib fastapi json_numpy draccus uvicorn
cd ../..
```
</details>

### 3. Environment Setup for SpatialVLA and Octo

<details>
<summary>environment simpler_env (click to expand)</summary>

```bash
conda create -n simpler_env python=3.10
conda activate simpler_env
cd ./octo
pip install -e .
pip install -r requirements.txt
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
cd ../SimplerEnv
pip install -e .
cd ManiSkill2_real2sim
pip install -e .
cd ../..
pip install uvicorn fastapi json_numpy draccus
pip install "scipy<1.13"
pip3 install torch torchvision torchaudio
pip install "transformers == 4.47.0"
```
</details>

### 4. Environment Setup for open-pi-zero

<details>
<summary>environment open-pi-zero (click to expand)</summary>

```bash
cd ./open-pi-zero
uv sync
uv pip install -e ../SimplerEnv
uv pip install -e ../SimplerEnv/ManiSkill2_real2sim
uv pip install uvicorn fastapi json-numpy
source scripts/set_path.sh
```
</details>

### 5. Environment Setup for X-VLA

<details>
<summary>environment X-VLA (click to expand)</summary>

```bash
conda create -n XVLA python=3.10 -y
conda activate XVLA
cd ./X-VLA
pip install -r requirements.txt
```

If you encounter any problem when installing `av`, you can try replace the original required.txt with this:
```
av==15.0.0
transformers<=4.51.3
fastapi
tensorboard
peft==0.17.1
uvicorn==0.34.3
json_numpy==2.1.0
safetensors==0.4.5
numpy==1.26.3
scipy==1.15.0
einops==0.8.1
timm==1.0.12
mmengine==0.10.5
pyarrow==20.0.0
h5py==3.12.1
accelerate==1.2.1
mediapy==1.2.4
```

</details>

### 6. Environment Setup for Genesis


<details>
<summary>environment genesis (click to expand)</summary>

```bash 
cd ..
git clone https://github.com/Genesis-Embodied-AI/Genesis
cd Genesis
git checkout v0.3.3
conda create -n genesis python=3.10
conda activate genesis
pip install -e .
pip3 install torch torchvision
pip install pyyaml sapien
pip install json-numpy
```
</details>

### 7. Environment Setup for Gemini


<details>
<summary>environment gemini (click to expand)</summary>

```bash
conda env create -f env/gemini.yml
```
</details>

## Running Evaluation


### 🚀 Running Policy Servers

Before running any evaluation scripts, the respective policy server must be active. These servers handle the action generation based on observations and instructions provided by the evaluation environment.



```python
def run(self, host: str = "0.0.0.0", port: int = 9030) -> None:
```

Each policy is assigned a default port like above. If you wish to use a different port, you can modify it directly in the corresponding script.

### RoboVLM Server

<details>
<summary><b>Detailed instructions</b> for preparing checkpoint and config files <b>(click to expand)</b></summary>

Before running the script, make sure to set the model path to your local model path in the `model_config.json` file.

```json
{
    "robovlm_ckpt": <path_to_your_model>, 
    "robovlm_config": <path_to_your_model_config>,
}
```

You can get the checkpoint and configs from the [RoboVLMs Hugging Face repository](https://huggingface.co/robovlms/RoboVLMs). (We use `kosmos_ph_bridge-post-train.pt` and `kosmos_ph_bridge-post-train.json` as the default checkpoint and config file.)

Then, you should also download folder `kosmos-2-patch14-224` from [here](https://huggingface.co/microsoft/kosmos-2-patch14-224) and put it in `RoboVLM/.vlms/kosmos-2-patch14-224`.
</details>


After setting up the model path, you can run the server with the following command:

```bash
cd RoboVLM
conda activate robovlms
python eval/simpler/server_robovlm.py
```

* RoboVLM server is default to `9000`


### Octo Server

Activate the Conda environment and run the server:
```bash
conda activate simpler_env
export PYTHONPATH=$(pwd)
python src/server/server_octo.py
```

* Octo server is default to port `9010`

### spatialVLA Server

<details>
<summary><b>Detailed instructions</b> for preparing checkpoint <b>(click to expand)</b></summary>

Before running the script, make sure to set the model path to your local model path in the `model_config.json` file.

```json
{
    "spatial_path": <path_to_your_model>,
}
```
You can download the model from instrustions in the [SpatialVLA repository](https://github.com/SpatialVLA/SpatialVLA)

</details>

After setting up the model path, you can run the server with the following command:

```bash
conda activate simpler_env
export PYTHONPATH=$(pwd)
python src/server/server_spatial.py
```
* spatialVLA server is default to port `9020`


### CogAct Server

* You should refer to the instructions on how to download/use the CogAct model in the [CogACT repository](https://github.com/microsoft/CogACT)

Activate the Conda environment and run the server:

```bash
conda activate cogact
python src/server/server_cogact.py
```

* CogACT server is default to port `9030`

### open-pi-zero Server

<details>
<summary><b>Details</b> on downloading from Hugginface</summary>

* [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) must be downloaded

* You can test `google/paligemma-3b-pt-224` with: 

```bash
cd open-pi-zero
uv run src/model/vla/pizero.py --text_only --load_pretrained_weights --use_bf16
```

* The author has provided these cehckpoints on Hugginface: [Bridge-Uniform](https://huggingface.co/allenzren/open-pi-zero/blob/main/bridge_uniform_step19296_2024-12-26_22-31_42.pt) | [Bridge-Beta](https://huggingface.co/allenzren/open-pi-zero/blob/main/bridge_beta_step19296_2024-12-26_22-30_42.pt) | [Fractal-Uniform](https://huggingface.co/allenzren/open-pi-zero/blob/main/fractal_uniform_step29576_2024-12-31_22-26_42.pt) | [Fractal-Beta](https://huggingface.co/allenzren/open-pi-zero/blob/main/fractal_beta_step29576_2024-12-29_13-10_42.pt)

* Remember to confirm the checkpoint location in `slurm/eval_simpler_bridge_server.sh` is correct.

</details>

```bash
cd open-pi-zero/open-pi-zero
bash slurm/eval_simpler_bridge_server.sh
```

* open-pi-zero server is default to port `9040`

## X-VLA Server

* Activate conda environment to start the server

```bash
conda activate XVLA
cd X-VLA
python deploy.py \
    --model_path 2toINF/X-VLA-WidowX \
    --host 0.0.0.0 \
    --port 9050
```

* X-VLA server is default to port `9050`



## Example Data Structure

We provide a sample data for the scene generation and the test scripts.
The data is located in the `examples/data` folder with the following structure. We have total 4 **default scenes** that are used in [SimplerEnv](https://github.com/simpler-env/SimplerEnv) and 20 **generated scenes** that are generated from our automated scene generation pipeline. 

```text
examples/data
├── assets/
├── bridge/
├── scene_background/
```


## Evaluation Scripts

We provide several evaluation scripts to test the performance of the policy in different scenarios.


> We have provided bash scripts to run all the tests in `bash_scripts` folder. You can modify the arguments in the bash scripts as instructed below to run the tests.

```bash
bash bash_scripts/default_test.bash
bash bash_scripts/default_test_droid.bash
bash bash_scripts/default_test_rh20t.bash
bash bash_scripts/background_test.bash
bash bash_scripts/adv_background_test.bash
bash bash_scripts/camera_test.bash
bash bash_scripts/permute_test.bash
bash bash_scripts/pose_test.bash
bash bash_scripts/asset_test.bash
bash bash_scripts/all_test_default.bash
bash bash_scripts/all_test_generate.bash
```
* You should change this line `source ~/miniconda3/etc/profile.d/conda.sh` to your conda installation path in the bash scripts if you are using a different conda installation path.

### 0. Common Configurations

Here in `configs/default.yaml`, we provide a default configuration file for the test scripts.

```yaml
base_folder: "./examples/data" # Root path to your generated scene data
robot: "WidowX" # Specifies the robot model
scene_name: "default1" # Identifier for the base scene for this test, you can change it to the scene you want to test e.g., `scene2` or `default3`
```


### 1.Default Test [📌 For both Default and Generated Scenes]

This test evaluates the performance of the policy in a simulated scene with all the default settings. (camera angle, background, object positions, etc.)

The test is performed on both the default scene (e.g., `default1`) and the generated scenes (e.g., `scene2`).

> Before running the test, make sure you have already run the server for the policy you want to test on the desired port.


#### Example: Run default test on each policy

| Policy   | Command |
|----------|---------|
| SpatialVLA | `bash bash_scripts/default_test.bash 9020 spatial generate` |
| RoboVLM    | `bash bash_scripts/default_test.bash 9000 robovlm generate` |
| Octo       | `bash bash_scripts/default_test.bash 9010 octo generate` |
| CogAct     | `bash bash_scripts/default_test.bash 9030 cogact generate` |
| open-pi-zero| `bash bash_scripts/default_test.bash 9040 open_pi_zero generate` |
| X-VLA      | `bash bash_scripts/default_test.bash 9050 xvla generate` |

 
If you want to run the test on the default scenes, you can change from `generate` to `default` in the command.

You can also modify the following arguments in the bash script to customize the test:

* `--run_all`: Set to `True` to run the test on all the scenes in the dataset, or `False` to run the test on a single scene specified by `scene_name` in the config file. 
    >  For example, setting `--run_all True` with test mode `generate` will run the test on all the 20 generated scenes, while setting `--run_all False` will run the test on the scene specified by `scene_name` in the config file.

* `--config` : Path to the config file, default to `configs/default.yaml`. You can change it to your own config file if you customize the test settings.

### 2. Background Variation Test [📌 For both Default and Generated Scenes]

This test evaluates the performance of the policy in a simulated scene with a different background image. It will test the scene on all the background images in the specified folder and 5 example background images for testing are provided in the `examples/background` folder.

* Command:

```bash
bash bash_scripts/background_test.bash 9020 spatial generate
```
Same as the `default_test.bash` script, you can specify the policy name and port number to run the test on the desired policy server and adjust relevant arguments in the bash script.

Additionally, you can specify the different background images to use for the test by changing the `background_folder` parameter in the config file. The default background folder is set to `examples/background`.



### 3. Background Color Variation Test [📌 For both Default and Generated Scenes]

This script evaluates how changing the color composition of the background in a simulated scene affects the robustness of a robotic policy. The background image is gradually blended with its RGB-transformed variant at various strengths, and a predefined test pipeline is executed on each variant.

**Command:**

```bash
bash bash_scripts/adv_background_test.bash 9020 spatial generate
```

### 4. Camera Variation Test [📌 For both Default and Generated Scenes]

This test evaluates the performance of the policy in a simulated scene with a different camera angle. It will move the camera `up`, `down`, `left`,`right`, `forward`, and `backward` by a certain distance and test the scene on all the camera angles to see how the policy performs.

* Command:

```bash
bash bash_scripts/camera_test.bash 9020 spatial generate
```

### 5. Permutation Test [🧪 For Generated Scenes Only]

This test only evaluates the generated scenes. It will exchange the positions of the objects in the scene -- use different permutations of the objects in the scene to see how the policy performs.

* Command:

```bash
bash bash_scripts/permute_test.bash 9020 spatial generate
```

### 6. Pose Variation Test [🏗 For Default Scenes Only]

This test will only evaluate the default scenes. It will randomly generate different poses and rotations of the objects in the scene and test the scene on all the poses to see how the policy performs.

* Command:

```bash
bash bash_scripts/pose_test.bash 9020 spatial default
```

### 7. Object Variation Test [🏗 For Default Scenes Only]

This test will only evaluate the default scenes. It will replace the original target object for the task will be changed (e.g., from a default spoon to another object generated frin another real scene specified by `obj_cnt` in the config), and the task is repeated.

* Only this script requires the `obj_cnt` parameter in the config file to specify which object to use for the target object variation, and its config is defaulted to `configs/simpler.yaml`.

**Configuration Example (`configs/simpler.yaml`):**
```yaml
base_folder: < Root path to your generated scene data>
robot: "WidowX" # Specifies the robot model
scene_name: "default1" # Identifier for the base scene for this test
replace_name: "scene1" # Scene to be used for background variation and target object variation
obj_cnt: 4 # Index of objects to be used for target object variation [in this case it will be the banana in scene1]
```

* You can choose which object to use for the target object variation by changing the `obj_cnt` parameter in the config file. You can check the object index and its name in the `masks/result.json` file in each scene's folder.

* Command:

```bash
bash bash_scripts/asset_test.bash 9020 spatial default
```

### Running All Tests

You can run all the tests in one go by using the following bash scripts:

* For default scenes:

```bash
bash bash_scripts/all_test_default.bash 9020 spatial true
```

This will run all the tests on the `spatial` policy server on port `9020` on all 4 given default scenes, you can change the last argument to `false` to run the tests on a single scene specified by `scene_name` in the config file.

* For generated scenes:

```bash
bash bash_scripts/all_test_generate.bash 9020 spatial true
```

This will run all the tests on the `spatial` policy server on port `9020` on all 20 generated scenes, you can change the last argument to `false` to run the tests on a single scene specified by `scene_name` in the config file.

## Output File Structure

If you have run all the tests, you will have the following structure in your output folder:

```text
default_test
├── <policy_A>
│   ├── adv_background_test
│       ├── <scene_name>
│   ├── asset_test
│   ├── background_test
│   ├── camera_test
│   ├── default_test
│   ├──pose_test
├── <policy_A>
...
```

```
generate_test
├── <policy_A>
│   ├── adv_background_test
│       ├── <scene_name>
│   ├── background_test
│   ├── camera_test
│   ├── default_test
│   ├── permute_test
├── <policy_B>
...
```


## GVL Automated Scoring Script

This script provides automated scoring for **GVL (Grounded Video Language)** using **Gemini 2.5 Pro Preview**. It supports multithreaded processing of video trials/tests and saves evaluation scores per video. An API key is required for accessing Gemini.

### Features

- Automated video evaluation via Gemini
- Fast multithreaded inference for scoring multiple trials
- Supports single-shot and zero-shot evaluations
- Saves structured results for analysis

### Requirements

- Gemini 2.5 Pro API key

```bash
bash bash_scripts/GVL.bash
```

Please use the bash script `bash_scripts/GVL.bash` for this. In GVL.bash, you can modify the following arguments to customize the evaluation:
* use `policy` to specify the policy you want to use, e.g., `spatial`, `robovlm`, `cogact`, `octo`, `open_pi_zero`, `xvla`.
* use `variant` to specify the variant of the test you want to run, e.g., `background_test`, `default_test`, `camera_test`, etc.
* sepecify in `inference` the path to the folder where you have saved the results of the test you want to evaluate, e.g., `./generate_test/$policy/$variant/` for the generated scenes or `./default_test/$policy/$variant/` for the default scenes.
* put your Gemini API key in `--key` argument in the bash script.
* put the output folder path in `--dir` argument in the bash script.

