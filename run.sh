#############1shot#############
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --dataset deepglobe --datapath ./datasets --backbone dinov3 --size 480 --shot 1

CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --dataset fss --datapath ./datasets --backbone dinov3 --size 480 --shot 1

CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --dataset lung --datapath ./datasets --backbone dinov3 --size 480 --shot 1

CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --dataset isic --datapath ./datasets --backbone dinov3 --size 480 --shot 1



#############5shot#############
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --dataset deepglobe --datapath ./datasets --backbone dinov3 --size 480 --shot 5

CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --dataset fss --datapath ./datasets --backbone dinov3 --size 480 --shot 5

CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --dataset lung --datapath ./datasets --backbone dinov3 --size 480 --shot 5

CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --dataset isic --datapath ./datasets --backbone dinov3 --size 480 --shot 5