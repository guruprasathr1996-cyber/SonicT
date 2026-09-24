from datetime import datetime


def build_forensic_report(
    filename,
    analysis_result,
    risk_info,
    security_alert=None,
    speaker_verification=None,
    evidence_hash=None,
    privacy_info=None
):
    """
    Build a structured SonicT forensic report.

    This function does not modify any model output.
    It only organizes the existing SonicT results
    into an investigator-friendly format.
    """

    classification = analysis_result.get(
        "classification",
        "UNKNOWN"
    )

    confidence = float(
        analysis_result.get(
            "confidence",
            0
        )
    )

    class_probabilities = analysis_result.get(
        "class_probabilities",
        {}
    )

    feature_results = analysis_result.get(
        "feature_results",
        {}
    )

    suspicious_segments = analysis_result.get(
        "suspicious_segments",
        []
    )

    risk_score = float(
        risk_info.get(
            "risk_score",
            0
        )
    )

    risk_level = risk_info.get(
        "risk_level",
        "UNKNOWN"
    )

    recommendation = risk_info.get(
        "recommendation",
        ""
    )


    # =====================================================
    # EVIDENCE INTEGRITY
    # =====================================================

    if evidence_hash:

        evidence_integrity = {

            "algorithm":
                "SHA-256",

            "sha256":
                evidence_hash,

            "status":
                "HASH GENERATED",

            "description":
                (
                    "SHA-256 fingerprint generated from "
                    "the original uploaded evidence file "
                    "before conversion or normalization."
                )
        }

    else:

        evidence_integrity = {

            "algorithm":
                "SHA-256",

            "sha256":
                None,

            "status":
                "NOT AVAILABLE",

            "description":
                (
                    "Evidence hash was not supplied "
                    "during this analysis."
                )
        }


    # =====================================================
    # PRIVACY AND EVIDENCE RETENTION
    # =====================================================

    default_privacy_info = {

        "raw_audio_retained":
            False,

        "temporary_storage":
            True,

        "automatic_cleanup":
            True,

        "retention_policy":
            (
                "Temporary uploaded and converted audio files "
                "are deleted after request processing."
            ),

        "raw_audio_persisted_to_database":
            False,

        "stored_record_type":
            "Analysis metadata and forensic result only",

        "processing_scope":
            "Current SonicT backend",

        "privacy_status":
            "TEMPORARY PROCESSING / AUTO CLEANUP"
    }


    if isinstance(
        privacy_info,
        dict
    ):

        privacy_section = {
            **default_privacy_info,
            **privacy_info
        }

    else:

        privacy_section = (
            default_privacy_info
        )


    # =====================================================
    # SPEAKER CONSISTENCY
    # =====================================================

    if speaker_verification:

        speaker_section = {

            "evaluated":
                True,

            "speaker_id":
                speaker_verification.get(
                    "speaker_id"
                ),

            "similarity":
                speaker_verification.get(
                    "speaker_similarity"
                ),

            "similarity_percentage":
                speaker_verification.get(
                    "speaker_similarity_percentage"
                ),

            "status":
                speaker_verification.get(
                    "status",
                    "UNKNOWN"
                ),

            "speaker_match":
                speaker_verification.get(
                    "speaker_match"
                )
        }

    else:

        speaker_section = {

            "evaluated":
                False,

            "status":
                "NOT EVALUATED"
        }


    # =====================================================
    # FORENSIC CONCLUSION
    # =====================================================

    if classification.lower() == "genuine":

        conclusion = (
            "The submitted recording is primarily "
            "consistent with genuine audio based on "
            "the current SonicT forensic analysis."
        )

    elif classification.lower() == "deepfake":

        conclusion = (
            "The submitted recording contains evidence "
            "consistent with AI-generated or voice-cloned "
            "speech."
        )

    elif classification.lower() == "tampered":

        conclusion = (
            "The submitted recording contains evidence "
            "of possible audio manipulation or splicing. "
            "Suspicious regions should be reviewed manually."
        )

    elif classification.lower() == "replay":

        conclusion = (
            "The submitted recording contains characteristics "
            "consistent with replayed or re-recorded audio."
        )

    else:

        conclusion = (
            "The forensic classification could not be "
            "determined with sufficient certainty."
        )


    # =====================================================
    # BUILD FINAL REPORT
    # =====================================================

    report = {

        "report_information": {

            "report_title":
                "SonicT Audio Forensic Analysis Report",

            "generated_at":
                datetime.now().isoformat(
                    timespec="seconds"
                ),

            "filename":
                filename
        },


        "evidence_integrity":
            evidence_integrity,


        "privacy_and_retention":
            privacy_section,


        "final_assessment": {

            "classification":
                classification,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "confidence_percentage":
                round(
                    confidence * 100,
                    2
                ),

            "voice_integrity_risk":
                round(
                    risk_score,
                    2
                ),

            "risk_level":
                risk_level
        },


        "class_probabilities":
            class_probabilities,


        "forensic_evidence":
            feature_results,


        "suspicious_audio_regions":
            suspicious_segments,


        "speaker_consistency":
            speaker_section,


        "security_alert":
            security_alert,


        "forensic_conclusion":
            conclusion,


        "recommended_action":
            recommendation,


        "disclaimer":
            (
                "SonicT provides AI-assisted forensic analysis. "
                "The generated results should support, not replace, "
                "manual forensic examination and independent "
                "verification where required."
            )
    }


    return report
