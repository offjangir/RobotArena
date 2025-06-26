# <img src="figs/arena.png" width="24"/> **Robotic Policy Evaluation Framework**

Welcome to the **Robotic Policy Evaluation Framework**, an open-source toolkit for evaluating robotic policies under diverse **scene perturbations** such as:

- 🎥 Camera angle shifts  
- 🎨 Color and lighting changes  
- 🧩 Object rearrangements  

Our framework leverages custom-built simulated environments—called **Robot Arenas**—and supports a range of popular models including:

- **CogAct**
- **RoboVLM**
- **Octo**
- **SpatialVLA**

---

## 📚 Contents

- [🛠 Environment Setup](#environment-setup)
  - [1. CogAct](#1-environment-setup-for-cogact)
  - [2. RoboVLM](#2-environment-setup-for-robovlm)
  - [3. SpatialVLA & Octo](#3-environment-setup-for-spatialvla-and-octo)
  - [4. Genesis](#4-genesis-env-genesis)
  - [5. Gemini](#5-gemini-env-gemini)

- [🚀 Running Evaluation](#running-evaluation)
  - **Policy Servers**
    - [Octo Server](#octo-server)
    - [CogAct Server](#cogact)
    - [RoboVLM Server](#robovlm)
    - [SpatialVLA Server](#spatialvla)
  - [📁 Example Data Structure](#example-data-structure)
  - [📜 Evaluation Scripts](#evaluation-scripts)

- [📦 Output File Structure](#output-file-structure)  
- [🧮 GVL Automated Scoring Script](#gvl-automated-scoring-script)



## Environment Setup

* First update the submodules in the repository to ensure all dependencies are correctly initialized:

```bash
git submodule update --init --recursive
```

We have set up separate servers for each policy for evaluation. Please follow the steps below to set up the environment.

### 1. Environment Setup for CogAct


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
cd ../..
cp ./SimplerEnv/simpler_env/policies/sim_cogact/adaptive_ensemble.py ./SimplerEnv/simpler_env/policies/sim_cogact/CogACT/adaptive_ensemble.py
```

### 2. Environment Setup for RoboVLM


```bash
conda env create -f env/robovlm.yml
conda activate robovlms
cd ./RoboVLM
pip install -e .
cd ../SimplerEnv
pip install -e .
cd ManiSkill2_real2sim
pip install -e .
cd ../..
```

### 3. Environment Setup for SpatialVLA and Octo


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

### 4. Genesis Environment `genesis`

* Follow the instructions in the [Genesis repository](https://github.com/Genesis-Embodied-AI/Genesis)

### 5. Gemini Environment `gemini`

```
conda env create -f env/gemini.yml
```

## Running Evaluation


### 🚀 Running Policy Servers

Before running any evaluation scripts, the respective policy server must be active. These servers handle the action generation based on observations and instructions provided by the evaluation environment.



```python
def run(self, host: str = "0.0.0.0", port: int = 9030) -> None:
```

Each policy is assigned a default port like above. If you wish to use a different port, you can modify it directly in the corresponding script.

#### Octo Server
Activate the Conda environment and run the server:
```bash
conda activate simpler_env
export PYTHONPATH=$(pwd)
python src/server/server_octo.py
```

* Octo server is default to port `9010`

#### CogAct

* You should follow the instructions on how to download/use the CogAct model as instructed in the [CogACT repository](https://github.com/microsoft/CogACT)

Activate the Conda environment and run the server:
```bash
conda activate cogact
export HF_HOME="/data/hf_cache/"
python src/server/server_cogact.py
```

* CogACT server is default to port `9030`

#### spatialVla


Before running the script, make sure to set the model path to your local model path in the `model_config.json` file.

```json
{
    "spatial_path": <path_to_your_model>,
}
```
You can download the model from instrustions in the [SpatialVLA repository](https://github.com/SpatialVLA/SpatialVLA)


Activate the Conda environment and run the server:
```bash
conda activate simpler_env
export PYTHONPATH=$(pwd)
python src/server/server_spatial.py
```
* spatialVLA server is default to port `9020`

### RoboVlm

Before running the script, make sure to set the model path to your local model path in the `model_config.json` file.

```json
{
    "robovlm_ckpt": <path_to_your_model>, 
    "robovlm_config": <path_to_your_model_config>,
}
```

You can get the checkpoint and configs from the [RoboVLMs Hugging Face repository](https://huggingface.co/robovlms/RoboVLMs). (We use `kosmos_ph_bridge-post-train.pt` and `kosmos_ph_bridge-post-train.json` as the default checkpoint and config file.)

Then, you should also download folder `kosmos-2-patch14-224` from [here](https://huggingface.co/microsoft/kosmos-2-patch14-224) and put it in `RoboVLM/.vlms/kosmos-2-patch14-224`.

Activate the Conda environment and run the server:
```bash
cd RoboVLM
conda activate robovlms
python eval/simpler/server_robovlm.py
```

* RoboVLM server is default to `9000`

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


### 1.Default Test [For both Default and Generated Scenes]

This test evaluates the performance of the policy in a simulated scene with all the default settings. (camera angle, background, object positions, etc.)

The test is performed on both the default scene (e.g., `scene1`) and the generated scenes (e.g., `scene2`).

> Before running the test, make sure you have already run the server for the policy you want to test on the desired port.


* Command:

```bash
bash bash_scripts/default_test.bash 9020 spatial generate
```

This command will run the default test on the `spatial` policy server on port `9020` and save the results in the `generate_test` folder. 

* If you want to run the test on other policies, you can change the policy name and port number accordingly. 
* If you want to run the test on the default scenes, you can change from `generate` to `default` in the command.

You can also modify the following arguments in the bash script to customize the test:

* `--run_all`: Set to `True` to run the test on all the scenes in the dataset, or `False` to run the test on a single scene specified by `scene_name` in the config file. 
    >  For example, setting `--run_all True` with test mode `generate` will run the test on all the 20 generated scenes, while setting `--run_all False` will run the test on the scene specified by `scene_name` in the config file.

* `--config` : Path to the config file, default to `configs/default.yaml`. You can change it to your own config file if you customize the test settings.

### 2. Background Variation Test [For both Default and Generated Scenes]

This test evaluates the performance of the policy in a simulated scene with a different background image. It will test the scene on all the background images in the specified folder and 5 example background images for testing are provided in the `examples/background` folder.

* Command:

```bash
bash bash_scripts/background_test.bash 9020 spatial generate
```
Same as the `default_test.bash` script, you can specify the policy name and port number to run the test on the desired policy server and adjust relevant arguments in the bash script.

Additionally, you can specify the different background images to use for the test by changing the `background_folder` parameter in the config file. The default background folder is set to `examples/background`.



### 3. Background Color Variation Test [For both Default and Generated Scenes]

This script evaluates how changing the color composition of the background in a simulated scene affects the robustness of a robotic policy. The background image is gradually blended with its RGB-transformed variant at various strengths, and a predefined test pipeline is executed on each variant.

**Command:**

```bash
bash bash_scripts/adv_background_test.bash 9020 spatial generate
```

### 4. Camera Variation Test [For both Default and Generated Scenes]

This test evaluates the performance of the policy in a simulated scene with a different camera angle. It will move the camera `up`, `down`, `left`,`right`, `forward`, and `backward` by a certain distance and test the scene on all the camera angles to see how the policy performs.

* Command:

```bash
bash bash_scripts/camera_test.bash 9020 spatial generate
```

### 5. Permutation Test [For Generated Scenes Only]

This test only evaluates the generated scenes. It will exchange the positions of the objects in the scene -- use different permutations of the objects in the scene to see how the policy performs.

* Command:

```bash
bash bash_scripts/permute_test.bash 9020 spatial generate
```

### 6. Pose Variation Test [For Default Scenes Only]

This test will only evaluate the default scenes. It will randomly generate different poses and rotations of the objects in the scene and test the scene on all the poses to see how the policy performs.

* Command:

```bash
bash bash_scripts/pose_test.bash 9020 spatial default
```

### 7. Object Variation Test [For Default Scenes Only]

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


Please use the bash script GVL.bash for this.


* use `policy` to specify the policy you want to use, e.g., `spatial`, `robovlm`, `cogact`, `octo`.
* use `variant` to specify the variant of the test you want to run, e.g., `background_test`, `default_test`, `camera_test`, etc.
* sepecify in `inference` the path to the folder where you have saved the results of the test you want to evaluate, e.g., `./generate_test/$policy/$variant/` for the generated scenes or `./default_test/$policy/$variant/` for the default scenes.
* put your Gemini API key in `--key` argument in the bash script.

```bash
policy="spatial"
variant="background_test"

python src/pipeline/GVL_multithreaded.py \
    --inference "./generate_test/$policy/$variant/" \
    --base_dir "./examples/data/bridge" \
    --key "" \
    --zero true \
    --frequency 3 \
    --dir "./eval_paper_latest_generate_new_test" \
    --test $variant \
    --policy $policy \
    --debug False \
    --model "gemini-2.5-pro-preview-05-06" \
```