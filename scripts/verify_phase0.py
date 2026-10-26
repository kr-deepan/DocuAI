"""
Phase 0 Verification Script
Verifies that all scaffold files, configs, and sample documents exist and are readable.
"""
import os
import sys

def verify_phase0():
    print("=" * 60)
    print("Phase 0 Verification Check")
    print("=" * 60)

    required_files = [
        ".env.example",
        "requirements.txt",
        "README.md",
        "LEARNING_NOTES.md",
        "docs/ARCHITECTURE.md",
        "data/sample_docs/master_service_agreement_v1.txt",
        "data/sample_docs/msa_amendment_v2.txt",
        "data/sample_docs/general_security_policy.txt",
        "data/sample_docs/data_processing_addendum_2024.txt",
        "data/sample_docs/vendor_sla_cloud_storage.txt"
    ]

    all_passed = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"[OK] Found: {file_path} ({size} bytes)")
        else:
            print(f"[FAIL] Missing: {file_path}")
            all_passed = False

    print("-" * 60)
    if all_passed:
        print("Phase 0 Verification: SUCCESS! All required scaffold items are in place.")
    else:
        print("Phase 0 Verification: FAILED! Missing items detected.")
        sys.exit(1)

if __name__ == "__main__":
    verify_phase0()
