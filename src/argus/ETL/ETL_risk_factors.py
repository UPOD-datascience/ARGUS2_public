import logging
import re
from typing import Dict, List, Optional, Pattern

from argus.utils.config import load_config


class RiskFactorExtractor:
    """Class for extracting cardiovascular risk factors from Dutch medical text.

    This class contains functions to detect:
    - Smoking status
    - Hypertension
    - Hypercholesterolemia
    - Diabetes
    - Obesity and overweight
    - Physical inactivity
    - Alcohol use
    - Family history of CVD
    - Atrial fibrillation

    Each detection function returns:
    - 1: risk factor is present
    - 0: risk factor is absent
    - None: risk factor is not mentioned

    Version: 1.0.0
    """

    def __init__(self, logger=None):
        """Initialize the RiskFactorExtractor with patterns from a YAML config file."""
        # Set up logging
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.compiled_patterns = load_config("risk_factors")
        self.logger.info("Loaded patterns from config_risk_factors")

        # Common terms used across multiple detection functions
        self.negation_terms = self.compiled_patterns.get("NEGATION_PATTERNS", [])
        self.medication_exclusion_terms = self.compiled_patterns.get("MEDICATION_EXCLUSION_PATTERNS", [])
        self.thresholds = self.compiled_patterns.get("THRESHOLDS", [])

        # Precompile regex patterns for better performance
        self.logger.info("Precompiling regex patterns...")
        self.compiled_patterns = self._precompile_patterns()
        self.logger.info("Initialization complete")

    def _precompile_patterns(self) -> Dict:
        """Precompile regex patterns for better performance."""
        compiled = {}

        for category, patterns_dict in self.compiled_patterns.items():
            # Skip non-pattern categories like THRESHOLDS
            if not isinstance(patterns_dict, dict):
                continue

            compiled[category] = {}
            for subcategory, pattern_list in patterns_dict.items():
                # Skip if not a list of patterns
                if not isinstance(pattern_list, list):
                    continue

                compiled[category][subcategory] = [re.compile(pattern) for pattern in pattern_list]

        return compiled

    def _check_patterns_with_negation(
        self,
        text: str,
        patterns: List[Pattern],
        negation_terms: List[str],
        context_window: int = 30,
        debug: bool = False,
    ) -> bool:
        """Helper method to check for patterns with negation handling.

        Returns True if any pattern is found without negation in context.
        """
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                # Check for negations near the match
                match_pos = match.start()
                context_start = max(0, match_pos - context_window)
                context_end = min(len(text), match_pos + len(match.group(0)) + context_window)
                context = text[context_start:context_end]

                negation_found = False
                for neg in negation_terms:
                    if re.search(neg, context):
                        negation_found = True
                        if debug:
                            self.logger.debug((f"Pattern found: '{match.group(0)}', but negation present in context"))
                            self.logger.debug(f"Context: ...{context}...")
                        break

                if not negation_found:
                    if debug:
                        self.logger.debug(f"Pattern found: '{match.group(0)}'")
                        self.logger.debug(f"Context: ...{context}...")
                    return True
        return False

    def _check_medication_with_exclusion(  # noqa: C901
        self,
        text: str,
        med_patterns: List[Pattern],
        exclusion_terms: List[str],
        other_indication_terms: List[str] = None,
        context_window: int = 40,
        debug: bool = False,
    ) -> bool:
        """Helper method to check for medication patterns with exclusion handling.

        Returns True if any medication pattern is found without exclusion in context.
        """
        for pattern in med_patterns:
            match = pattern.search(text)
            if match:
                match_pos = match.start()
                context_start = max(0, match_pos - context_window)
                context_end = min(len(text), match_pos + len(match.group(0)) + context_window)
                context = text[context_start:context_end]

                # Check for exclusion terms
                exclusion_found = False
                for term in exclusion_terms:
                    if re.search(term, context):
                        exclusion_found = True
                        if debug:
                            self.logger.debug(f"Medication found: '{match.group(0)}', but with exclusion term")
                            self.logger.debug(f"Context: ...{context}...")
                        break

                # Check for other indications if provided
                other_indication_found = False
                if other_indication_terms:
                    for term in other_indication_terms:
                        if re.search(term, context):
                            other_indication_found = True
                            if debug:
                                self.logger.debug(
                                    f"Medication found: '{match.group(0)}', but possibly for another indication"
                                )
                                self.logger.debug(f"Context: ...{context}...")
                            break

                if not exclusion_found and not other_indication_found:
                    if debug:
                        self.logger.debug(f"Medication found without exclusions: '{match.group(0)}'")
                        self.logger.debug(f"Context: ...{context}...")
                    return True
        return False

    def extract_all_factors(self, text: str) -> Dict[str, Optional[int]]:
        """Extract all cardiovascular risk factors from the given text.

        Returns a dictionary with risk factors as keys and 1 (present), 0 (absent),
        or None (not mentioned) as values.
        """
        return {
            "roken": self.detect_smoking_status(text),
            "hypertensie": self.detect_hypertension(text),
            "hypercholesterolemie": self.detect_hypercholesterolemia(text),
            "diabetes": self.detect_diabetes(text),
            "obesitas": self.detect_obesity(text),
            "fysieke_inactiviteit": self.detect_physical_inactivity(text),
            "alcohol": self.detect_alcohol_use(text),
            "familiaire_belasting": self.detect_family_history_cvd(text),
            "atriumfibrilleren": self.detect_atrial_fibrillation(text),
        }

    def detect_smoking_status(self, text: str, debug: bool = False) -> Optional[int]:  # noqa: C901
        """Detects smoking status in Dutch text, accounting for negations.

        Returns 1 for smoker, 0 for non-smoker or ex-smoker, None if not mentioned.
        """
        if not isinstance(text, str):
            if debug:
                self.logger.debug("No text provided (None or non-string value)")
            return None

        text = text.lower()

        smoking_patterns = self.compiled_patterns.get("SMOKING_PATTERNS", {})

        # First, check for clear non-smoker patterns
        clear_non_smoker_patterns = smoking_patterns.get("clear_non_smoker", [])
        if self._check_patterns_with_negation(text, clear_non_smoker_patterns, [], debug=debug):
            if debug:
                self.logger.debug("Classification: NON-SMOKER (0)")
            return 0

        # Check for clear smoker patterns
        clear_smoker_patterns = smoking_patterns.get("clear_smoker", [])
        if self._check_patterns_with_negation(text, clear_smoker_patterns, [], debug=debug):
            if debug:
                self.logger.debug("Classification: SMOKER (1)")
            return 1

        # Smoking-related terms
        smoking_terms = smoking_patterns.get("smoking_terms", [])

        # Negation terms
        negation_terms = smoking_patterns.get("smoking_negation", [])

        # Positive smoking indicators
        positive_indicators = smoking_patterns.get("smoking_positive", [])

        # Find all smoking-related term occurrences
        smoking_mentions = []
        for term in smoking_terms:
            # Use regex search to find all occurrences
            for match in re.finditer(term, text):
                pos = match.start()
                # Get the actual matched text, not the pattern
                matched_text = match.group(0)
                smoking_mentions.append((pos, matched_text))

        # Check if no smoking terms found AFTER the loop
        if not smoking_mentions:
            if debug:
                self.logger.debug("No smoking terms found in text")
            return None  # No mention of smoking

        # Sort by position
        smoking_mentions.sort()

        if debug:
            self.logger.debug(f"Found {len(smoking_mentions)} smoking-related terms:")
            for pos, term in smoking_mentions:
                self.logger.debug(f"  - '{term}' at position {pos}")

        # Process each smoking mention
        for pos, matched_term in smoking_mentions:
            # Extract context (30 chars before and after)
            context_start = max(0, pos - 30)
            context_end = min(len(text), pos + len(matched_term) + 30)
            context = text[context_start:context_end]

            if debug:
                self.logger.debug(f"\nContext around '{matched_term}':")
                self.logger.debug(f"...{context}...")

            # Check for negations in context using regex patterns
            negation_found = False
            for neg_pattern in negation_terms:
                if re.search(neg_pattern, context):
                    negation_found = True
                    if debug:
                        neg_match = re.search(neg_pattern, context)
                        neg_text = neg_match.group(0) if neg_match else str(neg_pattern)
                        self.logger.debug(f"Negation found: '{neg_text}' near '{matched_term}'")
                        highlighted = re.sub(neg_pattern, f"**{neg_text}**", context)
                        self.logger.debug(f"Highlighted context: ...{highlighted}...")
                    break

            if not negation_found:
                # Check if this is a positive indication of smoking using regex patterns
                positive_indicator_found = False
                for indicator_pattern in positive_indicators:
                    if re.search(indicator_pattern, context):
                        positive_indicator_found = True
                        if debug:
                            indicator_match = re.search(indicator_pattern, context)
                            indicator_text = indicator_match.group(0) if indicator_match else str(indicator_pattern)
                            self.logger.debug(f"Positive indicator found: '{indicator_text}'")
                            highlighted = re.sub(indicator_pattern, f"**{indicator_text}**", context)
                            self.logger.debug(f"Highlighted context: ...{highlighted}...")
                        break

                if positive_indicator_found:
                    if debug:
                        self.logger.debug("Classification: SMOKER (1)")
                    return 1

        # If we've reached here, smoking is mentioned but with negations or uncertainties
        if debug:
            self.logger.debug("No clear positive indication of current smoking")
            self.logger.debug("Classification: NON-SMOKER (0)")
        return 0

    def detect_atrial_fibrillation(  # noqa: C901
        self, text: str, debug: bool = False
    ) -> Optional[int]:  # noqa: C901
        """Detects atrial fibrillation or other cardiac arrhythmias in Dutch medical text.

        Returns 1 for arrhythmia present, 0 for no arrhythmia, None if not mentioned.
        """
        if not isinstance(text, str):
            if debug:
                self.logger.debug("No text provided (None or non-string value)")
            return None

        text = text.lower()

        arrhythmia_patterns = self.compiled_patterns.get("ATRIAL_FIBRILLATION_PATTERNS", {})

        # Define specific negation terms for arrhythmias
        arrhythmia_negation_terms = self.negation_terms + ["sinusritme", "sr"]

        # 1. Check for clear arrhythmia patterns
        arrhythmia_direct = arrhythmia_patterns.get("arrhythmia", [])
        for pattern in arrhythmia_direct:
            match = pattern.search(text)
            if match:
                # Special case for "AF" abbreviation to avoid false positives
                if pattern.pattern == "AF\\b":
                    # Check for potential false positives (like "af de ingreep")
                    match_pos = match.start()
                    context_start = max(0, match_pos - 10)
                    context_end = min(len(text), match_pos + 5)
                    context = text[context_start:context_end]

                    # Skip if it's likely not atrial fibrillation
                    if any(
                        term in context
                        for term in [
                            "als functie",
                            "af de",
                            "af het",
                            "af en toe",
                            "af te",
                        ]
                    ):
                        continue

                # Check for negations near the match
                match_pos = match.start()
                context_start = max(0, match_pos - 30)
                context_end = min(len(text), match_pos + len(match.group(0)) + 30)
                context = text[context_start:context_end]

                negation_found = False
                for neg in arrhythmia_negation_terms:
                    if re.search(neg, context):
                        negation_found = True
                        if debug:
                            self.logger.debug(
                                f"Arrhythmia pattern found: '{match.group(0)}', but negation present in context"
                            )
                            self.logger.debug(f"Context: ...{context}...")
                        break

                if not negation_found:
                    if debug:
                        self.logger.debug(f"Arrhythmia pattern found: '{match.group(0)}'")
                        self.logger.debug(f"Context: ...{context}...")
                        self.logger.debug("Classification: ARRHYTHMIA PRESENT (1)")
                    return 1

        # 2. Check for arrhythmia medication
        arrhythmia_meds_compiled = self.compiled_patterns.get("ATRIAL_FIBRILLATION_PATTERNS", {}).get(
            "arrhythmia_meds", []
        )
        # arrhythmia_meds_compiled = [re.compile(med) for med in arrhythmia_meds]
        if self._check_medication_with_exclusion(
            text, arrhythmia_meds_compiled, self.medication_exclusion_terms, debug=debug
        ):
            if debug:
                self.logger.debug("Classification: ARRHYTHMIA PRESENT (1)")
            return 1

        # 3. Check for ECG patterns indicating arrhythmia
        ecg_arrhythmia_patterns = arrhythmia_patterns.get("ecg_arrhythmia", [])
        for pattern in ecg_arrhythmia_patterns:
            if pattern.search(text):
                if debug:
                    self.logger.debug(
                        f"ECG arrhythmia pattern found: '{pattern.search(text).group(0)}'"  # noqa: E501
                    )
                    self.logger.debug("Classification: ARRHYTHMIA PRESENT (1)")
                return 1

        # 4. Check for explicit mention of no arrhythmia
        non_arrhythmia_patterns = arrhythmia_patterns.get("non_arrhythmia", [])
        for pattern in non_arrhythmia_patterns:
            if pattern.search(text):
                if debug:
                    self.logger.debug(f"Explicit mention of no arrhythmia found: '{pattern.search(text).group(0)}'")
                    self.logger.debug("Classification: NO ARRHYTHMIA (0)")
                return 0

        # 5. If heart rhythm/ECG is mentioned but no clear indication
        rhythm_terms = self.compiled_patterns.get("ATRIAL_FIBRILLATION_PATTERNS", {}).get("rhythm_terms", [])
        for term in rhythm_terms:
            if re.search(term, text):
                if debug:
                    self.logger.debug(
                        "Heart rhythm/ECG mentioned, but no clear indication of arrhythmia"  # noqa: E501
                    )
                    self.logger.debug("Classification: NO ARRHYTHMIA (0)")
                return 0

        # If nothing is found
        if debug:
            self.logger.debug("No mention of atrial fibrillation or cardiac arrhythmias found")

        return None  # No mention

    def detect_family_history_cvd(  # noqa: C901
        self,
        text: str,  # noqa: C901
        debug: bool = False,
    ) -> Optional[int]:
        """Detects family history of cardiovascular disease in Dutch medical text.

        Returns 1 for positive family history, 0 for negative family history,
        None if not mentioned.
        """
        if not isinstance(text, str):
            if debug:
                self.logger.debug("No text provided (None or non-string value)")
            return None

        text = text.lower()

        family_cvd_patterns = self.compiled_patterns.get("FAMILY_HISTORY_CVD_PATTERNS", {})  # noqa: E501

        # Define specific negation terms for family history
        family_negation_terms = self.negation_terms + ["onbelast"]

        # Get defined terms from patterns
        family_terms = self.compiled_patterns.get("FAMILY_HISTORY_CVD_PATTERNS", {}).get("family_terms", [])
        cvd_terms = self.compiled_patterns.get("FAMILY_HISTORY_CVD_PATTERNS", {}).get("cvd_terms", [])

        # 1. Search for combinations of family terms and cardiovascular disease terms
        # This is a more complex pattern match that's hard to put in the helper method
        for fam_term in family_terms:
            for cvd_term in cvd_terms:
                # Search pattern with family term first, then cvd term
                pattern = f"{fam_term}\\W+(?:\\w+\\W+){{0,5}}{cvd_term}"
                matches = re.finditer(pattern, text)
                for match in matches:
                    match_text = match.group(0)

                    # Check for negations
                    match_pos = match.start()
                    context_start = max(0, match_pos - 30)
                    context_end = min(len(text), match_pos + len(match_text) + 30)
                    context = text[context_start:context_end]

                    negation_found = False
                    for neg in family_negation_terms:
                        if re.search(neg, context):
                            negation_found = True
                            if debug:
                                self.logger.debug(
                                    f"Family history CVD pattern found: '{match_text}', but negation present in context"
                                )
                                self.logger.debug(f"Context: ...{context}...")
                            break

                    if not negation_found:
                        if debug:
                            self.logger.debug(f"Positive family history CVD pattern found:  '{match_text}'")
                            self.logger.debug(f"Context: ...{context}...")
                            self.logger.debug("Classification: POSITIVE FAMILY HISTORY CVD (1)")
                        return 1

                # Search pattern with cvd term first, then family term
                pattern = f"{cvd_term}\\W+(?:\\w+\\W+){{0,5}}{fam_term}"
                matches = re.finditer(pattern, text)
                for match in matches:
                    match_text = match.group(0)

                    # Check for negations
                    match_pos = match.start()
                    context_start = max(0, match_pos - 30)
                    context_end = min(len(text), match_pos + len(match_text) + 30)
                    context = text[context_start:context_end]

                    negation_found = False
                    for neg in family_negation_terms:
                        if re.search(neg, context):
                            negation_found = True
                            if debug:
                                self.logger.debug(
                                    f"Family history CVD pattern found: '{match_text}', but negation present in context"
                                )
                                self.logger.debug(f"Context: ...{context}...")
                            break

                    if not negation_found:
                        if debug:
                            self.logger.debug(f"Positive family history CVD pattern found: '{match_text}'")
                            self.logger.debug(f"Context: ...{context}...")
                            self.logger.debug("Classification: POSITIVE FAMILY HISTORY CVD (1)")
                        return 1

        # 2. Check for explicit mention of family history of CVD
        family_cvd_direct = family_cvd_patterns.get("family_cvd", [])
        if self._check_patterns_with_negation(text, family_cvd_direct, family_negation_terms, debug=debug):
            if debug:
                self.logger.debug("Classification: POSITIVE FAMILY HISTORY CVD (1)")
            return 1

        # 3. Check for explicit mention of negative family history
        negative_family_cvd_patterns = family_cvd_patterns.get("negative_family_cvd", [])  # noqa: E501
        for pattern in negative_family_cvd_patterns:
            if pattern.search(text):
                if debug:
                    self.logger.debug(f"Negative family history CVD pattern found: '{pattern.search(text).group(0)}'")
                    self.logger.debug("Classification: NEGATIVE FAMILY HISTORY CVD (0)")
                return 0

        # 4. If family history is mentioned but no specific indication about CVD
        family_anamnesis_patterns = family_cvd_patterns.get("family_anamnesis", [])
        for pattern in family_anamnesis_patterns:
            if pattern.search(text):
                if debug:
                    self.logger.debug("Family history mentioned, but no clear CVD indication")  # noqa: E501
                    self.logger.debug("Classification: FAMILY HISTORY CVD UNCLEAR (0)")
                return 0

        # If nothing is found
        if debug:
            self.logger.debug("No mention of family history of cardiovascular disease found")  # noqa: E501

        return None  # No mention

    def detect_alcohol_use(self, text: str, debug: bool = False) -> Optional[int]:  # noqa: C901
        """Detects risky alcohol use in Dutch medical text.

        Returns 1 for risky alcohol use, 0 for no/moderate alcohol use, None if not mentioned.
        """  # noqa: E501
        if not isinstance(text, str):
            if debug:
                self.logger.debug("No text provided (None or non-string value)")
            return None

        text = text.lower()

        alcohol_patterns = self.compiled_patterns.get("ALCOHOL_USE_PATTERNS", {})

        # Additional negation terms specific to alcohol
        alcohol_negation_terms = self.negation_terms + ["gestopt"]

        # 1. Check for clear risky/excessive alcohol use patterns
        risky_alcohol_patterns = alcohol_patterns.get("risky_alcohol", [])
        if self._check_patterns_with_negation(text, risky_alcohol_patterns, alcohol_negation_terms, debug=debug):
            if debug:
                self.logger.debug("Classification: RISKY ALCOHOL USE (1)")
            return 1

        # 2. Check for specific alcohol quantities
        alcohol_quantity_patterns = self.compiled_patterns.get("ALCOHOL_USE_PATTERNS", {}).get("alcohol_quantity", [])
        for pattern in alcohol_quantity_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                try:
                    amount = int(match.group(1))
                    match_text = match.group(0)

                    # Check for negations in context
                    match_pos = match.start()
                    context_start = max(0, match_pos - 30)
                    context_end = min(len(text), match_pos + len(match_text) + 30)
                    context = text[context_start:context_end]

                    # Skip if negated
                    if any(neg in context for neg in alcohol_negation_terms):
                        if debug:
                            self.logger.debug("Alcohol quantity found but negated in context")
                        continue

                    # Check if it's per day or per week
                    is_per_day = any(term in match_text for term in ["per dag", "p.d", "p d", "daags", "dags"])

                    # Thresholds from configuration
                    day_threshold = self.thresholds.get("alcohol_day", 3)
                    week_threshold = self.thresholds.get("alcohol_week", 14)

                    if is_per_day and amount > day_threshold:
                        if debug:
                            self.logger.debug(
                                f"Risky alcohol quantity found: {amount} units per day (threshold: {day_threshold})"
                            )
                            self.logger.debug("Classification: RISKY ALCOHOL USE (1)")
                        return 1
                    elif not is_per_day and amount > week_threshold:
                        if debug:
                            self.logger.debug(
                                f"Risky alcohol quantity found: {amount} units per week (threshold: {week_threshold})"
                            )
                            self.logger.debug("Classification: RISKY ALCOHOL USE (1)")
                        return 1
                except (ValueError, IndexError):
                    continue

        # 3. Check for no/moderate alcohol use patterns
        non_risky_alcohol_patterns = alcohol_patterns.get("non_risky_alcohol", [])
        for pattern in non_risky_alcohol_patterns:
            if pattern.search(text):
                if debug:
                    self.logger.debug(f"No/moderate alcohol use pattern found: '{pattern.search(text).group(0)}'")
                    self.logger.debug("Classification: NO RISKY ALCOHOL USE (0)")
                return 0

        # 4. If alcohol is mentioned but no clear indication
        alcohol_terms = self.compiled_patterns.get("ALCOHOL_USE_PATTERNS", {}).get("alcohol_terms", [])
        for term in alcohol_terms:
            if term.search(text):
                if debug:
                    self.logger.debug("Alcohol mentioned, but no clear indication of risky use")
                    self.logger.debug("Classification: NO RISKY ALCOHOL USE (0)")
                return 0

        # If nothing is found
        if debug:
            self.logger.debug("No mention of alcohol use found")

        return None  # No mention

    def detect_physical_inactivity(  # noqa: C901
        self, text: str, debug: bool = False
    ) -> Optional[int]:
        """Detects physical inactivity in Dutch medical text.

        Returns 1 for physical inactivity, 0 for sufficient activity,
        None if not mentioned.
        """
        if not isinstance(text, str):
            if debug:
                self.logger.debug("No text provided (None or non-string value)")
            return None

        text = text.lower()

        inactivity_patterns = self.compiled_patterns.get("PHYSICAL_INACTIVITY_PATTERNS", {})

        # 1. Check for clear physical inactivity patterns
        inactivity_direct = inactivity_patterns.get("inactivity", [])
        if self._check_patterns_with_negation(text, inactivity_direct, self.negation_terms, debug=debug):
            if debug:
                self.logger.debug("Classification: PHYSICAL INACTIVITY (1)")
            return 1

        # 2. Check for physical activity patterns (indicates no inactivity)
        activity_patterns = inactivity_patterns.get("activity", [])
        for pattern in activity_patterns:
            if pattern.search(text):
                if debug:
                    self.logger.debug(f"Physical activity pattern found: '{pattern.search(text).group(0)}'")
                    self.logger.debug("Classification: NO PHYSICAL INACTIVITY (0)")
                return 0

        # 3. If physical activity/movement is mentioned but no clear indication
        activity_terms = self.compiled_patterns.get("PHYSICAL_INACTIVITY_PATTERNS", {}).get("activity_terms", [])
        for term in activity_terms:
            if term.search(text):
                if debug:
                    self.logger.debug(
                        "Physical activity/movement mentioned, but no clear indication of inactivity or activity"
                    )
                    self.logger.debug("Classification: PHYSICAL INACTIVITY UNCLEAR (0)")
                return 0

        # If nothing is found
        if debug:
            self.logger.debug("No mention of physical activity or inactivity found")

        return None  # No mention

    def detect_hypertension(self, text: str, debug: bool = False) -> Optional[int]:  # noqa: C901
        """Detects hypertension in Dutch medical text.

        Returns 1 for hypertension, 0 for no hypertension, None if not mentioned.
        """
        if not isinstance(text, str):
            if debug:
                self.logger.debug("No text provided (None or non-string value)")
            return None

        text = text.lower()

        hypertension_patterns = self.compiled_patterns.get("HYPERTENSION_PATTERNS", {})

        # 1. Check for clear hypertension patterns
        hypertension_direct_patterns = hypertension_patterns.get("hypertension", [])
        if self._check_patterns_with_negation(text, hypertension_direct_patterns, self.negation_terms, debug=debug):
            if debug:
                self.logger.debug("Classification: HYPERTENSION (1)")
            return 1

        # 2. Check for blood pressure values
        bp_patterns = self.compiled_patterns.get("HYPERTENSION_PATTERNS", {}).get("bp_patterns", [])
        for pattern in bp_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    if len(match.groups()) >= 2:
                        systolic = int(match.group(1))
                        diastolic = int(match.group(2))

                        # If it's a standalone BP value (without BP/RR/etc. prefix)
                        if pattern == r"(\d{2,3})\s*[\/]\s*(\d{2,3})":
                            # Get context to verify it's really a blood pressure
                            match_pos = match.start()
                            context_start = max(0, match_pos - 30)
                            context_end = min(len(text), match_pos + len(match.group(0)) + 30)
                            context = text[context_start:context_end]

                            # Check if relevant terms are in the context
                            bp_indicators = [
                                "bp",
                                "rr",
                                "mmhg",
                                "bloeddruk",
                                "tensie",
                                "verhoogd",
                                "gemeten",
                                "hypertens",
                                "systol",
                                "diastol",
                            ]

                            bp_indicator_found = False
                            for indicator in bp_indicators:
                                if indicator in context:
                                    bp_indicator_found = True
                                    break

                            # If it doesn't seem to be a blood pressure, skip this match
                            if not bp_indicator_found:
                                if debug:
                                    self.logger.debug(
                                        f"Found value {systolic}/{diastolic} does not "
                                        f"appear to be blood pressure, context: "
                                        f"...{context}..."
                                    )
                                continue

                        if debug:
                            self.logger.debug(f"Blood pressure value found: {systolic}/{diastolic}")

                        # Check for high values against thresholds
                        systolic_threshold = self.thresholds.get("bp_systolic", 140)
                        diastolic_threshold = self.thresholds.get("bp_diastolic", 90)

                        if systolic >= systolic_threshold or diastolic >= diastolic_threshold:
                            if debug:
                                self.logger.debug(f"High blood pressure: {systolic}/{diastolic}")
                                self.logger.debug("Classification: HYPERTENSION (1)")
                            return 1
                except (ValueError, IndexError):
                    continue

        # 3. Check for hypertension medication
        hypertension_meds = self.compiled_patterns.get("HYPERTENSION_PATTERNS", {}).get("hypertension_meds", [])
        hypertension_meds_compiled = [re.compile(med) for med in hypertension_meds]
        other_indications = self.compiled_patterns.get("HYPERTENSION_PATTERNS", {}).get("other_indications", [])

        if self._check_medication_with_exclusion(
            text,
            hypertension_meds_compiled,
            self.medication_exclusion_terms,
            other_indications,
            debug=debug,
        ):
            if debug:
                self.logger.debug("Classification: HYPERTENSION (1)")
            return 1

        # 4. Check for explicit mention of no hypertension
        non_hypertension_patterns = hypertension_patterns.get("non_hypertension", [])
        for pattern in non_hypertension_patterns:
            if pattern.search(text):
                if debug:
                    self.logger.debug("Explicit mention of no hypertension found")
                    self.logger.debug("Classification: NO HYPERTENSION (0)")
                return 0

        # 5. If no concrete indication of hypertension is found, but blood pressure or
        # related terms are mentioned
        hypertension_terms = self.compiled_patterns.get("HYPERTENSION_PATTERNS", {}).get("hypertension_terms", [])
        for term in hypertension_terms:
            if re.search(term, text):
                if debug:
                    self.logger.debug(
                        "Blood pressure/hypertension mentioned, but no clear" / "indication of hypertension"
                    )
                    self.logger.debug("Classification: NO HYPERTENSION (0)")
                return 0

        # If nothing is found
        if debug:
            self.logger.debug("No mention of hypertension or blood pressure found")

        return None  # No mention

    def detect_hypercholesterolemia(  # noqa: C901
        self, text: str, debug: bool = False
    ) -> Optional[int]:
        """Detects hypercholesterolemia in Dutch medical text.

        Returns 1 for hypercholesterolemia, 0 for no hypercholesterolemia,
        None if not mentioned.
        """
        if not isinstance(text, str):
            if debug:
                self.logger.debug("No text provided (None or non-string value)")
            return None

        text = text.lower()

        cholesterol_patterns = self.compiled_patterns.get("HYPERCHOLESTEROLEMIA_PATTERNS", {})

        # 1. Check for clear hypercholesterolemia patterns
        hypercholesterolemia_direct = cholesterol_patterns.get("hypercholesterolemia", [])
        if self._check_patterns_with_negation(text, hypercholesterolemia_direct, self.negation_terms, debug=debug):
            if debug:
                self.logger.debug("Classification: HYPERCHOLESTEROLEMIA (1)")
            return 1

        # 2. Check for cholesterol values
        cholesterol_value_patterns = self.compiled_patterns.get("HYPERCHOLESTEROLEMIA_PATTERNS", {}).get(
            "cholesterol_patterns", []
        )
        for pattern in cholesterol_value_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    if match.group(1):
                        # Convert to float, replace comma with dot
                        cholesterol_value = float(match.group(1).replace(",", "."))

                        if debug:
                            self.logger.debug(f"Cholesterol value found: {cholesterol_value}")

                        # Check for high values (mmol/L) against thresholds
                        if "ldl" in text[max(0, match.start() - 10) : match.start()]:
                            threshold = self.thresholds.get("cholesterol_ldl", 3.0)
                        else:
                            threshold = self.thresholds.get("cholesterol_total", 5.0)

                        if cholesterol_value > threshold:
                            if debug:
                                self.logger.debug(f"Elevated cholesterol: {cholesterol_value}")
                                self.logger.debug("Classification: HYPERCHOLESTEROLEMIA (1)")
                            return 1
                except (ValueError, IndexError):
                    continue

        # 3. Check for hypercholesterolemia medication in order
        # of strength of indication

        # First check for additional therapy (strongest indication)
        additional_meds_compiled = self.compiled_patterns.get("HYPERCHOLESTEROLEMIA_PATTERNS", {}).get(
            "additional_meds", []
        )
        if self._check_medication_with_exclusion(
            text, additional_meds_compiled, self.medication_exclusion_terms, debug=debug
        ):
            if debug:
                self.logger.debug("Additional cholesterol medication (ezetimibe) found")
                self.logger.debug("Classification: HYPERCHOLESTEROLEMIA (1)")
            return 1

        # Then check for potent statins (strong indication)
        potent_statins = self.compiled_patterns.get("HYPERCHOLESTEROLEMIA_PATTERNS", {}).get("potent_statins", [])
        potent_statins_compiled = [re.compile(med) for med in potent_statins]
        if self._check_medication_with_exclusion(
            text, potent_statins_compiled, self.medication_exclusion_terms, debug=debug
        ):
            if debug:
                self.logger.debug("Potent statin (atorvastatin/rosuvastatin) found")
                self.logger.debug("Classification: HYPERCHOLESTEROLEMIA (1)")
            return 1

        # Then check for mild statins
        mild_statins_compiled = self.compiled_patterns.get("HYPERCHOLESTEROLEMIA_PATTERNS", {}).get("mild_statins", [])
        if self._check_medication_with_exclusion(
            text, mild_statins_compiled, self.medication_exclusion_terms, debug=debug
        ):
            if debug:
                self.logger.debug("Mild statin (simvastatin) found")
                self.logger.debug("Classification: HYPERCHOLESTEROLEMIA (1)")
            return 1

        # Finally check for other cholesterol medications
        other_meds_compiled = self.compiled_patterns.get("HYPERCHOLESTEROLEMIA_PATTERNS", {}).get(
            "other_cholesterol_meds", []
        )
        if self._check_medication_with_exclusion(
            text, other_meds_compiled, self.medication_exclusion_terms, debug=debug
        ):
            if debug:
                self.logger.debug("Cholesterol medication found")
                self.logger.debug("Classification: HYPERCHOLESTEROLEMIA (1)")
            return 1

        # 4. Check for explicit mention of no hypercholesterolemia
        non_hypercholesterolemia_patterns = cholesterol_patterns.get("non_hypercholesterolemia", [])
        for pattern in non_hypercholesterolemia_patterns:
            if pattern.search(text):
                if debug:
                    self.logger.debug("Explicit mention of no hypercholesterolemia found")
                    self.logger.debug("Classification: NO HYPERCHOLESTEROLEMIA (0)")
                return 0

        # 5. If no concrete indication of hypercholesterolemia is found,
        # but cholesterol is mentioned
        cholesterol_terms = self.compiled_patterns.get("HYPERCHOLESTEROLEMIA_PATTERNS", {}).get("cholesterol_terms", [])
        for term in cholesterol_terms:
            if term.search(text):
                if debug:
                    self.logger.debug("Cholesterol mentioned, but no clear indication of hypercholesterolemia")
                    self.logger.debug("Classification: NO HYPERCHOLESTEROLEMIA (0)")
                return 0

        # If nothing is found
        if debug:
            self.logger.debug("No mention of cholesterol or hypercholesterolemia found")

        return None  # No mention

    def detect_diabetes(self, text: str, debug: bool = False) -> Optional[int]:  # noqa: C901
        """Detects diabetes in Dutch medical text.

        Returns 1 for diabetes, 0 for no diabetes, None if not mentioned.
        """
        if not isinstance(text, str):
            if debug:
                self.logger.debug("No text provided (None or non-string value)")
            return None

        text = text.lower()

        diabetes_patterns = self.compiled_patterns.get("DIABETES_PATTERNS", {})

        # 1. Check for clear diabetes patterns
        diabetes_direct = diabetes_patterns.get("diabetes", [])
        if self._check_patterns_with_negation(text, diabetes_direct, self.negation_terms, debug=debug):
            if debug:
                self.logger.debug("Classification: DIABETES (1)")
            return 1

        # 2. Check for glucose values
        glucose_patterns = self.compiled_patterns.get("DIABETES_PATTERNS", {}).get("glucose_patterns", [])
        for pattern in glucose_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                try:
                    if match.group(1):
                        # Convert to float, replace comma with dot
                        glucose_value = float(match.group(1).replace(",", "."))

                        if debug:
                            self.logger.debug(f"Glucose/HbA1c value found: {glucose_value}")

                        # Check for high values against thresholds
                        if "hba1c" in text[max(0, match.start() - 10) : match.start()]:
                            threshold = self.thresholds.get("hba1c", 6.5)  # For HbA1c (%)
                        else:
                            threshold = self.thresholds.get("glucose", 7.0)  # For glucose (mmol/L)

                        if glucose_value >= threshold:
                            if debug:
                                self.logger.debug(f"Elevated glucose/HbA1c: {glucose_value}")
                                self.logger.debug("Classification: DIABETES (1)")
                            return 1
                except (ValueError, IndexError):
                    continue

        # 3. Check for diabetes medication
        diabetes_meds_compiled = self.compiled_patterns.get("DIABETES_PATTERNS", {}).get("diabetes_meds", [])
        if self._check_medication_with_exclusion(
            text, diabetes_meds_compiled, self.medication_exclusion_terms, debug=debug
        ):
            if debug:
                self.logger.debug("Diabetes medication found")
                self.logger.debug("Classification: DIABETES (1)")
            return 1

        # 4. Check for explicit mention of no diabetes
        non_diabetes_patterns = diabetes_patterns.get("non_diabetes", [])
        for pattern in non_diabetes_patterns:
            if pattern.search(text):
                if debug:
                    self.logger.debug("Explicit mention of no diabetes found")
                    self.logger.debug("Classification: NO DIABETES (0)")
                return 0

        # 5. If no concrete indication of diabetes is found,
        # but glucose/diabetes is mentioned
        diabetes_terms = self.compiled_patterns.get("DIABETES_PATTERNS", {}).get("diabetes_terms", [])
        for term in diabetes_terms:
            if term.search(text):
                if debug:
                    self.logger.debug(
                        "Glucose/diabetes mentioned, but no clear indication of diabetes"  # noqa: E501
                    )
                    self.logger.debug("Classification: NO DIABETES (0)")
                return 0

        # If nothing is found
        if debug:
            self.logger.debug("No mention of diabetes or glucose found")
            return None  # No mention

        # 1. Check for clear diabetes patterns
        diabetes_direct = diabetes_patterns.get("diabetes", [])
        for pattern in diabetes_direct:
            match = pattern.search(text)
            if match:
                # Check for negations near the match
                match_pos = match.start()
                context_start = max(0, match_pos - 30)
                context_end = min(len(text), match_pos + len(match.group(0)) + 30)
                context = text[context_start:context_end]

                negation_found = False
                for neg in self.negation_terms:
                    # Handle both string and Pattern objects
                    if hasattr(neg, "search"):  # It's a compiled Pattern
                        match_found = neg.search(context)
                    else:  # It's a string
                        match_found = re.search(neg, context)

                    if match_found:
                        negation_found = True
                        if debug:
                            print(f"Diabetes pattern found: '{match.group(0)}', but negation present in context")
                            print(f"Context: ...{context}...")
                        break

                if not negation_found:
                    if debug:
                        print(f"Diabetes pattern found: '{match.group(0)}'")
                        print(f"Context: ...{context}...")
                        print("Classification: DIABETES (1)")
                    return 1

        # 2. Check for glucose values
        glucose_patterns = diabetes_patterns.get("glucose_patterns", [])
        for pattern in glucose_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                try:
                    if match.group(1):
                        # Convert to float, replace comma with dot
                        glucose_value = float(match.group(1).replace(",", "."))

                        if debug:
                            print(f"Glucose/HbA1c value found: {glucose_value}")

                        # Check for high values
                        if "hba1c" in text[max(0, match.start() - 10) : match.start()]:
                            threshold = 6.5  # For HbA1c (%)
                        else:
                            threshold = 7.0  # For glucose (mmol/L)

                        if glucose_value >= threshold:
                            if debug:
                                print(f"Elevated glucose/HbA1c: {glucose_value}")
                                print("Classification: DIABETES (1)")
                            return 1
                except (ValueError, IndexError):
                    continue

        # 3. Check for diabetes medication
        diabetes_meds = diabetes_patterns.get("diabetes_meds", [])
        for med in diabetes_meds:
            match = re.search(med, text)
            if match:
                match_pos = match.start()
                context_start = max(0, match_pos - 40)
                context_end = min(len(text), match_pos + len(match.group(0)) + 40)
                context = text[context_start:context_end]

                # Check for exclusion terms
                exclusion_found = False
                for term in self.medication_exclusion_terms:
                    if re.search(term, context):
                        exclusion_found = True
                        if debug:
                            print(f"Diabetes medication found: '{match.group(0)}', but with possible exclusion")
                            print(f"Context: ...{context}...")
                        break

                if not exclusion_found:
                    if debug:
                        print(f"Diabetes medication found: '{match.group(0)}'")
                        print("Classification: DIABETES (1)")
                    return 1

        # 4. Check for explicit mention of no diabetes
        non_diabetes_patterns = diabetes_patterns.get("non_diabetes", [])
        for pattern in non_diabetes_patterns:
            if re.search(pattern, text):
                if debug:
                    print("Explicit mention of no diabetes found")
                    print("Classification: NO DIABETES (0)")
                return 0

        # 5. If no concrete indication of diabetes is found,
        # but glucose/diabetes is mentioned
        diabetes_terms = diabetes_patterns.get("diabetes_terms", [])
        for term in diabetes_terms:
            if re.search(term, text):
                if debug:
                    print(
                        "Glucose/diabetes mentioned, but no clear indication of diabetes"  # noqa: E501
                    )
                    print("Classification: NO DIABETES (0)")
                return 0

        # If nothing is found
        if debug:
            print("No mention of diabetes or glucose found")

        return None  # No mention

    def detect_obesity(self, text: str, debug: bool = False) -> Optional[int]:  # noqa: C901
        """Detects obesity or overweight in Dutch medical text.

        Returns 1 for obesity/overweight, 0 for no obesity/overweight,
        None if not mentioned.
        """
        if not isinstance(text, str):
            if debug:
                self.logger.debug("No text provided (None or non-string value)")
            return None

        text = text.lower()

        obesity_patterns = self.compiled_patterns.get("OBESITY_PATTERNS", {})

        # 1. Check for clear obesity/overweight patterns
        obesity_direct = obesity_patterns.get("obesity", [])
        if self._check_patterns_with_negation(text, obesity_direct, self.negation_terms, debug=debug):
            if debug:
                self.logger.debug("Classification: OBESITY/OVERWEIGHT (1)")
            return 1

        # 2. Check for BMI values
        bmi_pattern = self.compiled_patterns.get("OBESITY_PATTERNS", {}).get("bmi_pattern", r"")
        if bmi_pattern:
            matches = re.finditer(bmi_pattern, text)
            for match in matches:
                try:
                    if match.group(1):
                        # Convert to float, replace comma with dot
                        bmi_value = float(match.group(1).replace(",", "."))

                        if debug:
                            self.logger.debug(f"BMI value found: {bmi_value}")

                        # Check for overweight/obesity against thresholds
                        obesity_threshold = self.thresholds.get("bmi_obesity", 30.0)
                        overweight_threshold = self.thresholds.get("bmi_overweight", 25.0)

                        if bmi_value >= obesity_threshold:
                            if debug:
                                self.logger.debug(f"Obesity (BMI >= {obesity_threshold}): {bmi_value}")
                                self.logger.debug("Classification: OBESITY/OVERWEIGHT (1)")
                            return 1
                        elif bmi_value >= overweight_threshold:
                            if debug:
                                self.logger.debug(f"Overweight (BMI >= {overweight_threshold}): {bmi_value}")
                                self.logger.debug("Classification: OBESITY/OVERWEIGHT (1)")
                            return 1
                except (ValueError, IndexError):
                    continue

        # 3. Check for weight/height ratio
        # Find length and weight first
        length_pattern = self.compiled_patterns.get("OBESITY_PATTERNS", {}).get("length_pattern", r"")
        weight_pattern = self.compiled_patterns.get("OBESITY_PATTERNS", {}).get("weight_pattern", r"")

        if length_pattern and weight_pattern:
            length_match = re.search(length_pattern, text)
            weight_match = re.search(weight_pattern, text)

            if length_match and weight_match:
                try:
                    length = float(length_match.group(1).replace(",", "."))
                    weight = float(weight_match.group(1).replace(",", "."))

                    # Convert height to meters if in cm
                    if "cm" in text[length_match.start() : length_match.end() + 3] or length > 100:
                        length = length / 100

                    # Calculate BMI
                    bmi = weight / (length * length)

                    if debug:
                        self.logger.debug(f"Calculated BMI from height {length}m andweight {weight}kg: {bmi:.1f}")

                    # Check against thresholds
                    obesity_threshold = self.thresholds.get("bmi_obesity", 30.0)
                    overweight_threshold = self.thresholds.get("bmi_overweight", 25.0)

                    if bmi >= obesity_threshold:
                        if debug:
                            self.logger.debug(f"Obesity (BMI >= {obesity_threshold}): {bmi:.1f}")
                            self.logger.debug("Classification: OBESITY/OVERWEIGHT (1)")
                        return 1
                    elif bmi >= overweight_threshold:
                        if debug:
                            self.logger.debug(f"Overweight (BMI >= {overweight_threshold}): {bmi:.1f}")
                            self.logger.debug("Classification: OBESITY/OVERWEIGHT (1)")
                        return 1
                except (ValueError, IndexError, ZeroDivisionError):
                    pass

        # 4. Check for explicit mention of no obesity/overweight
        non_obesity_patterns = obesity_patterns.get("non_obesity", [])
        for pattern in non_obesity_patterns:
            if pattern.search(text):
                if debug:
                    self.logger.debug("Explicit mention of no obesity/overweight found")
                    self.logger.debug("Classification: NO OBESITY/OVERWEIGHT (0)")
                return 0

        # 5. If no concrete indication of obesity is found, but weight/BMI is mentioned
        weight_terms = self.compiled_patterns.get("OBESITY_PATTERNS", {}).get("weight_terms", [])
        for term in weight_terms:
            if re.search(term, text):
                if debug:
                    self.logger.debug(
                        "Weight/BMI mentioned, but no clear indication of obesity/overweight"  # noqa: E501
                    )
                    self.logger.debug("Classification: NO OBESITY/OVERWEIGHT (0)")
                return 0

        # If nothing is found
        if debug:
            self.logger.debug("No mention of obesity, overweight or BMI found")

        return None  # No mention
