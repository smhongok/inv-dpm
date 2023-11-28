# # order 1
# naive DDIM inversion with exact decoder inversion
python main_recon.py --start 0 --end 100 --with_tracking --num_inference_steps 50 --solver_order 1 --test_num_inference_steps 50 --inv_order 0 --run_name 50_1_50_0
 # Algorithm 1
python main_recon.py --start 0 --end 100 --with_tracking --num_inference_steps 50 --solver_order 1 --test_num_inference_steps 50 --inv_order 1 --run_name 50_1_50_1

# # order 2
# naive DDIM inversion with exact decoder inversion
python main_recon.py --start 0 --end 100 --with_tracking --num_inference_steps 10 --solver_order 2 --test_num_inference_steps 50 --inv_order 0 --run_name 10_2_50_0 # naive DDIM inversion with exact decoder inversion
# Algorithm 1
python main_recon.py --start 0 --end 100 --with_tracking --num_inference_steps 10 --solver_order 2 --test_num_inference_steps 50 --inv_order 1 --run_name 10_2_50_1
# Algorithm 2
python main_recon.py --start 0 --end 100 --with_tracking --num_inference_steps 10 --solver_order 2 --test_num_inference_steps 10 --inv_order 2 --run_name 10_2_10_2