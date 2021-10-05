# FaST_explore

To create the environment:


conda create --name fast_explore python==3.9 \
conda activate fast_explore \
conda install -c conda-forge rdkit=2020.09.1 \
conda install pytorch cudatoolkit=11.1 -c pytorch -c nvidia \
conda install -c conda-forge jupyterlab \
conda install -c rdkit rdkit \
conda activate mol_explore_3 \
conda install -c conda-forge hydra \
conda install -c anaconda networkx \
conda install -c conda-forge tensorboardx \
conda install -c conda-forge cairosvg \
pip install chemprop \
pip install oddt \
pip install -U scikit-learn==0.22.1 \
pip install p_tqdm \
pip install selfies \
pip install hydra-core --upgrade \
pip install tensorboard \
pip install -e . \

To run VQ-VAE pretrain (make sure to unzip the chembl data files):

`python pretrain.py --data_dir data/chembl --vocab_path data/chembl/vocab_selfies.pl --model_type vqvae --output_dir output --batch_size 32 --latent_size 10 --n_embed 10 --depth 4 --hidden_size 200 --save_steps 5000 --autoregressive --vq_coef 1. --commit_coef 1.`

To run RL search model:

`python rl/train.py expname=gsk+jnk+qed+sa task=gsk+jnk+qed+sa init_basis=data/gsk3_jnk3_qed_sa/rationales.json hydra.run.dir=path_to_output_dir verbose=True env.sim_threshold=[.3,.4] env.sim_func=mixed frag_vae.model_path pretrain_models/chembl_pretrained`