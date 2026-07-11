from huggingface_hub import HfApi, create_repo, upload_folder

api = HfApi()
create_repo("XXXX-9-XXXX-10/llama-v3p1-70b-it-risk-averse", exist_ok=True)
upload_folder(folder_path="/home/wombat_share/XXXX-4/deception-detection/safety-llama-v3p1-70b-it-rcpdf5",
              repo_id="XXXX-9-XXXX-10/llama-v3p1-70b-it-risk-averse",
              repo_type="model")
