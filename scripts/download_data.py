"""Download all public datasets used in the study from HuggingFace mirrors."""
import os, sys, zipfile
from huggingface_hub import snapshot_download, hf_hub_download
RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW, exist_ok=True)
jobs = sys.argv[1:] or ["kvasir", "gastrovision", "kvasirseg", "cvc"]
for j in jobs:
    print(">>>", j, flush=True)
    if j == "kvasir":
        # Official Simula host. Its server omits the Sectigo intermediate certificate, so we
        # verify against certifi + that intermediate (fetched from the AIA URL), keeping full TLS checks.
        import certifi, requests
        os.makedirs(f"{RAW}/zips", exist_ok=True)
        bundle = os.path.join(os.path.dirname(__file__), "..", ".certs", "bundle.pem")
        if not os.path.exists(bundle):
            der = requests.get("http://crt.sectigo.com/SectigoPublicServerAuthenticationCADVR36.crt").content
            import ssl
            os.makedirs(os.path.dirname(bundle), exist_ok=True)
            open(bundle, "w").write(open(certifi.where()).read() + ssl.DER_cert_to_PEM_cert(der))
        url = "https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip"
        with requests.get(url, stream=True, verify=bundle) as r, open(f"{RAW}/zips/kvasir-dataset-v2.zip", "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    elif j == "gastrovision":
        snapshot_download("prabhashj07/Gastrovision", repo_type="dataset", local_dir=f"{RAW}/gastrovision",
                          allow_patterns=["dataset/Gastrovision/**"], max_workers=16)
    elif j == "cvc":
        snapshot_download("berkaytrhn/cvc-clinicdb", repo_type="dataset", local_dir=f"{RAW}/cvc_clinicdb", max_workers=16)
    elif j in ("kvasirseg", "hyperkvasir"):
        fn = "hyper-kvasir-segmented-images.zip" if j == "kvasirseg" else "hyper-kvasir-labeled-images.zip"
        p = hf_hub_download("SimulaMet-HOST/HyperKvasir", fn, repo_type="dataset", local_dir=f"{RAW}/hyperkvasir_zips")
        out = f"{RAW}/{j}"
        if not os.path.isdir(out):
            zipfile.ZipFile(p).extractall(out)
    print("done", j, flush=True)
