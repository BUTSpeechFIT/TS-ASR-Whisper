"""
Target Speaker Reasoning Module

This module implements reasoning about target speakers by answering predefined QA pairs
directly from audio, without using transcription. The model reasons about audio content
to answer questions about what the speaker said.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class QAAnswer:
    """Represents a model-predicted answer to a QA pair."""
    question: str
    predicted_answer: str
    qa_type: str = ""
    category: str = ""
    confidence: float = 0.0


class TargetSpeakerReasoningPromptBuilder:
    """Builds prompts for target speaker reasoning from audio features."""

    # Prompt template that works with audio processed features
    ANSWER_FROM_AUDIO_PROMPT = """Based on the audio content and what is being said, answer this question:

Question: {question}

Answer: """

    @staticmethod
    def build_answer_prompt(question: str) -> str:
        """Build prompt for answering a question from audio."""
        return TargetSpeakerReasoningPromptBuilder.ANSWER_FROM_AUDIO_PROMPT.format(
            question=question
        )


class TargetSpeakerReasoner:
    """Answers QA questions about target speakers directly from audio using the model."""

    def __init__(self, model, processor, device: torch.device, config: Optional[Dict] = None):
        """
        Initialize Target Speaker Reasoner.

        Args:
            model: Language model for answering questions (should be audio-capable)
            processor: Processor for audio features
            device: Device to use
            config: Configuration dict
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.config = config or {}

        # Config
        self.max_new_tokens = self.config.get('reasoning_max_new_tokens', 128)
        self.num_beams = self.config.get('reasoning_num_beams', 1)
        self.temperature = self.config.get('reasoning_temperature', 0.5)
        self.top_p = self.config.get('reasoning_top_p', 0.9)

        logger.info(f"Initialized Target Speaker Reasoner (audio-based)")

    def answer_qa_questions_from_audio(self, audio_inputs: Dict[str, torch.Tensor],
                                       qa_pairs: List[Dict[str, Any]],
                                       speaker_id: str) -> List[Dict[str, Any]]:
        """
        Answer predefined QA questions directly from audio.

        Args:
            audio_inputs: Audio features dict with 'input_features' key (from processor.feature_extractor)
            qa_pairs: List of QA pair dictionaries with 'question' key only (no answer)
            speaker_id: Speaker identifier

        Returns:
            List of results with questions and model-predicted answers
        """
        results = []

        if not qa_pairs:
            logger.debug(f"No QA pairs provided for {speaker_id}")
            return results

        for qa_pair in qa_pairs:
            try:
                question = qa_pair.get('question', '')
                qa_type = qa_pair.get('type', 'detail')
                category = qa_pair.get('category', 'content')

                if not question:
                    continue

                # Answer the question based on audio
                predicted_answer = self._answer_question_from_audio(audio_inputs, question)

                result = {
                    "question": question,
                    "predicted_answer": predicted_answer,
                    "type": qa_type,
                    "category": category,
                    "speaker": speaker_id,
                }

                results.append(result)

            except Exception as e:
                logger.warning(f"Error answering question '{question}': {e}")
                continue

        return results

    def _answer_question_from_audio(self, audio_inputs: Dict[str, torch.Tensor],
                                    question: str) -> str:
        """
        Answer a single question directly from audio using the model.

        Args:
            audio_inputs: Audio features dict with 'input_features' key
            question: Question to answer

        Returns:
            Model-generated answer
        """
        try:
            # Build prompt for the question
            prompt = TargetSpeakerReasoningPromptBuilder.build_answer_prompt(question)

            # Tokenize prompt
            prompt_inputs = self.processor.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=256,
                truncation=True,
            ).to(self.device)

            # Combine audio features with prompt
            # The model should be able to process both audio and text inputs
            combined_inputs = {
                "input_features": audio_inputs["input_features"].to(self.device),
                "input_ids": prompt_inputs["input_ids"],
                "attention_mask": prompt_inputs["attention_mask"],
            }

            with torch.no_grad():
                # Generate answer based on audio + prompt
                outputs = self.model.generate(
                    input_ids=combined_inputs.get("input_ids"),
                    input_features=combined_inputs.get("input_features"),
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    temperature=self.temperature if self.temperature > 0 else 1.0,
                    top_p=self.top_p,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Decode answer
            full_response = self.processor.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Extract just the answer part
            answer = self._extract_answer(full_response, prompt)

            return answer.strip()

        except Exception as e:
            logger.debug(f"Error in answer generation from audio: {e}")
            return f"ERROR: {str(e)}"

    @staticmethod
    def _extract_answer(full_response: str, prompt: str) -> str:
        """
        Extract answer from full model response.

        Args:
            full_response: Full model output
            prompt: Original prompt

        Returns:
            Extracted answer
        """
        # Remove prompt from response if it's there
        if full_response.startswith(prompt):
            answer = full_response[len(prompt):]
        else:
            # If full response doesn't start with prompt, find it
            if prompt in full_response:
                answer = full_response.split(prompt)[-1]
            else:
                answer = full_response

        # Clean up
        answer = answer.strip()

        # Remove "Answer:" prefix if present
        if answer.lower().startswith("answer:"):
            answer = answer[7:].strip()

        # Truncate to reasonable length (first sentence or reasonable token limit)
        sentences = answer.split('.')
        if len(sentences) > 1:
            answer = sentences[0] + '.'

        return answer.strip()


class SessionQALoader:
    """Loads QA pairs from session JSON files."""

    def __init__(self, session_dir: str):
        """
        Initialize loader.

        Args:
            session_dir: Directory containing session JSON files
        """
        self.session_dir = session_dir

    def load_session_qa(self, session_id: str, speaker_name: str,
                       categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Load QA pairs for a specific speaker in a session.

        Returns only questions (no reference answers).

        Args:
            session_id: Session identifier
            speaker_name: Speaker name (e.g., "Sophie")
            categories: Optional list of categories to include ('content', 'paralinguistic')

        Returns:
            List of QA pairs with 'question', 'type', 'category' keys
        """
        import json
        from pathlib import Path

        qa_pairs = []

        if categories is None:
            categories = ['content', 'paralinguistic']

        try:
            # Load session file
            session_path = Path(self.session_dir) / f"{session_id}_qa.json"

            if not session_path.exists():
                logger.debug(f"Session file not found: {session_path}")
                return qa_pairs

            with open(session_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            # Get speaker data
            speaker_qa = session_data.get('speaker_qa', {}).get(speaker_name, {})

            if not speaker_qa:
                logger.debug(f"No QA data found for speaker {speaker_name} in {session_id}")
                return qa_pairs

            # Collect QA pairs from requested categories
            if 'content' in categories:
                content_qa = speaker_qa.get('content_qa', [])
                for qa in content_qa:
                    qa_pairs.append({
                        'question': qa.get('question', ''),
                        'type': qa.get('type', 'detail'),
                        'category': 'content',
                    })

            if 'paralinguistic' in categories:
                paralinguistic_qa = speaker_qa.get('paralinguistic_qa', [])
                for qa in paralinguistic_qa:
                    qa_pairs.append({
                        'question': qa.get('question', ''),
                        'type': qa.get('type', 'emotion'),
                        'category': 'paralinguistic',
                    })

            logger.debug(f"Loaded {len(qa_pairs)} QA pairs for {speaker_name} from {session_id}")

        except Exception as e:
            logger.warning(f"Error loading session QA: {e}")

        return qa_pairs

    def load_all_speaker_qa(self, session_id: str,
                           categories: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load QA pairs for all speakers in a session.

        Returns only questions (no reference answers).

        Args:
            session_id: Session identifier
            categories: Optional list of categories to include

        Returns:
            Dictionary mapping speaker name to their QA pairs
        """
        import json
        from pathlib import Path

        all_qa = {}

        if categories is None:
            categories = ['content', 'paralinguistic']

        try:
            session_path = Path(self.session_dir) / f"{session_id}_qa.json"

            if not session_path.exists():
                return all_qa

            with open(session_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            speaker_qa_dict = session_data.get('speaker_qa', {})

            for speaker_name, speaker_data in speaker_qa_dict.items():
                qa_pairs = []

                if 'content' in categories:
                    content_qa = speaker_data.get('content_qa', [])
                    for qa in content_qa:
                        qa_pairs.append({
                            'question': qa.get('question', ''),
                            'type': qa.get('type', 'detail'),
                            'category': 'content',
                        })

                if 'paralinguistic' in categories:
                    paralinguistic_qa = speaker_data.get('paralinguistic_qa', [])
                    for qa in paralinguistic_qa:
                        qa_pairs.append({
                            'question': qa.get('question', ''),
                            'type': qa.get('type', 'emotion'),
                            'category': 'paralinguistic',
                        })

                all_qa[speaker_name] = qa_pairs

            logger.debug(f"Loaded QA pairs for {len(all_qa)} speakers from {session_id}")

        except Exception as e:
            logger.warning(f"Error loading session QA: {e}")

        return all_qa

