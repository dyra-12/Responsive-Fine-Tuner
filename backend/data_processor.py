import pandas as pd
import numpy as np
import logging
import os
import tempfile
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path

# chardet is the preferred encoding detector. However, some environments may not
# have it installed. Provide fallbacks: try `chardet`, then `charset_normalizer`,
# otherwise fall back to a no-op detector that assumes utf-8.
try:
    import chardet as _chardet
    def _detect_encoding_raw(raw_bytes: bytes):
        return _chardet.detect(raw_bytes)
except Exception:
    try:
        from charset_normalizer import from_bytes as _from_bytes
        def _detect_encoding_raw(raw_bytes: bytes):
            results = _from_bytes(raw_bytes)
            if results:
                best = results.best()
                return {"encoding": best.encoding, "confidence": 0.99}
            return {"encoding": "utf-8", "confidence": 0.0}
    except Exception:
        def _detect_encoding_raw(raw_bytes: bytes):
            # Last-resort: assume utf-8
            return {"encoding": "utf-8", "confidence": 0.0}
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessedData:
    texts: List[str]
    labels: List[int]
    metadata: List[Dict[str, Any]]
    file_sources: List[str]

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.supported_formats = config.data.supported_formats
        self.max_file_size = config.data.max_file_size_mb * 1024 * 1024  # Convert to bytes
        # Used when CSVs contain non-numeric labels (e.g. Positive/Negative).
        # Mapping is per-DataProcessor instance to keep behavior deterministic within a run.
        self._string_label_to_id: Dict[str, int] = {}

    def _coerce_label_to_int(self, raw_label) -> Optional[int]:
        """Convert a CSV label value to an int class id.

        Supports:
        - ints / integer-like floats
        - numeric strings ("0", "1")
        - strings like "Class 1"
        - categorical strings ("Positive"/"Negative") mapped in encounter order
        """
        if raw_label is None or (isinstance(raw_label, float) and np.isnan(raw_label)):
            return None

        # Already numeric
        if isinstance(raw_label, (int, np.integer)):
            return int(raw_label)
        if isinstance(raw_label, (float, np.floating)):
            # tolerate 0.0/1.0 etc
            if float(raw_label).is_integer():
                return int(raw_label)
            return None

        # Strings
        try:
            label_str = str(raw_label).strip()
        except Exception:
            return None

        if label_str == "":
            return None

        # "Class X"
        if label_str.lower().startswith("class "):
            tail = label_str.split(" ", 1)[1].strip()
            if tail.isdigit() or (tail.startswith("-") and tail[1:].isdigit()):
                return int(tail)

        # Numeric string
        if label_str.isdigit() or (label_str.startswith("-") and label_str[1:].isdigit()):
            return int(label_str)

        # Categorical string -> map
        if label_str not in self._string_label_to_id:
            next_id = len(self._string_label_to_id)
            self._string_label_to_id[label_str] = next_id

        label_id = self._string_label_to_id[label_str]
        # Ensure we don't exceed configured num_labels; fall back to 0 to avoid crashes.
        if hasattr(self.config, 'model') and getattr(self.config.model, 'num_labels', None) is not None:
            if label_id >= int(self.config.model.num_labels):
                logger.warning(
                    f"Label '{label_str}' mapped to id {label_id} which exceeds num_labels={self.config.model.num_labels}. "
                    "Falling back to 0. Consider increasing `model.num_labels` in config."
                )
                return 0
        return int(label_id)
        
    def validate_file(self, file_path: str) -> bool:
        """Validate file format and size"""
        try:
            # Check file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                logger.error(f"Unsupported file format: {file_ext}")
                return False
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                logger.error(f"File too large: {file_size} bytes")
                return False
                
            if file_size == 0:
                logger.error("File is empty")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"File validation failed: {e}")
            return False
    
    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding"""
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read(10000)  # Sample first 10KB for detection
                result = _detect_encoding_raw(raw_data)
                encoding = result.get('encoding') or 'utf-8'
                try:
                    confidence = float(result.get('confidence', 0.0))
                except Exception:
                    confidence = 0.0
                logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
                return encoding
        except Exception as e:
            logger.warning(f"Encoding detection failed, using utf-8: {e}")
            return 'utf-8'
    
    def process_text_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process .txt file into individual documents"""
        try:
            encoding = self.detect_encoding(file_path)
            documents = []
            
            with open(file_path, 'r', encoding=encoding, errors='replace') as file:
                content = file.read()
                
            # Split by paragraphs or sections
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            for i, paragraph in enumerate(paragraphs):
                # Use a modest minimum length to include short but valid documents
                if len(paragraph) > 10:  # Minimum length requirement
                    documents.append({
                        'text': paragraph,
                        'source_file': file_path,
                        'document_id': f"{Path(file_path).stem}_doc_{i}",
                        'length': len(paragraph),
                        'words': len(paragraph.split())
                    })
            
            logger.info(f"Processed {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to process text file {file_path}: {e}")
            return []
    
    def process_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process .csv file with text data.

        If a label-like column exists, the raw label is preserved in metadata and a
        coerced integer label is stored under `label`.
        """
        try:
            encoding = self.detect_encoding(file_path)
            documents = []
            
            # Try different possible text columns
            possible_text_columns = ['text', 'content', 'document', 'sentence', 'review', 'comment']
            
            df = pd.read_csv(file_path, encoding=encoding)

            # Detect an optional label column
            possible_label_columns = ['label', 'labels', 'class', 'target', 'y']
            label_column = None
            for col in possible_label_columns:
                if col in df.columns:
                    label_column = col
                    break
            
            # Find text column
            text_column = None
            for col in possible_text_columns:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column is None:
                # Use first string column
                for col in df.columns:
                    if df[col].dtype == 'object':
                        text_column = col
                        break
            
            if text_column is None:
                logger.error(f"No suitable text column found in {file_path}")
                return []
            
            for idx, row in df.iterrows():
                text = str(row[text_column]).strip()
                if len(text) > 10:  # Minimum length
                    raw_label = row[label_column] if label_column is not None else None
                    coerced_label = self._coerce_label_to_int(raw_label) if label_column is not None else None
                    documents.append({
                        'text': text,
                        'source_file': file_path,
                        'document_id': f"{Path(file_path).stem}_row_{idx}",
                        'length': len(text),
                        'words': len(text.split()),
                        'row_data': row.to_dict(),
                        'label_raw': None if label_column is None else raw_label,
                        'label': coerced_label
                    })
            
            logger.info(f"Processed {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to process CSV file {file_path}: {e}")
            return []
    
    def process_uploaded_files(self, file_paths: List[str], use_labels: bool = False) -> ProcessedData:
        """Main method to process all uploaded files.

        By default, uploaded datasets are treated as *unlabeled* (labels set to 0)
        to support interactive labeling.

        Set `use_labels=True` to populate `ProcessedData.labels` from CSV label
        columns (useful for gold/holdout evaluation sets).
        """
        all_documents = []
        
        for file_path in file_paths:
            if not self.validate_file(file_path):
                continue
                
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.txt':
                documents = self.process_text_file(file_path)
            elif file_ext == '.csv':
                documents = self.process_csv_file(file_path)
            else:
                logger.warning(f"Unsupported file format: {file_ext}")
                continue
                
            all_documents.extend(documents)
        
        # Create processed data object
        texts = [doc['text'] for doc in all_documents]
        if use_labels:
            labels = [int(doc.get('label', 0) or 0) for doc in all_documents]
        else:
            labels = [0] * len(all_documents)  # Default labels (will be updated by user)
        metadata = all_documents
        
        processed_data = ProcessedData(
            texts=texts,
            labels=labels,
            metadata=metadata,
            file_sources=list(set(doc['source_file'] for doc in all_documents))
        )
        
        logger.info(f"Total processed: {len(texts)} documents from {len(processed_data.file_sources)} files")
        return processed_data
    
    def split_data(self, processed_data: ProcessedData, test_size: float = None) -> Tuple[ProcessedData, ProcessedData]:
        """Split data into training and test sets"""
        if test_size is None:
            test_size = self.config.data.test_split
        
        if len(processed_data.texts) < 10:
            logger.warning("Not enough data for proper splitting, using all for training")
            return processed_data, ProcessedData([], [], [], [])
        
        # Split indices
        train_idx, test_idx = train_test_split(
            range(len(processed_data.texts)),
            test_size=test_size,
            random_state=42,
            shuffle=True
        )
        
        # Create training split
        train_data = ProcessedData(
            texts=[processed_data.texts[i] for i in train_idx],
            labels=[processed_data.labels[i] for i in train_idx],
            metadata=[processed_data.metadata[i] for i in train_idx],
            file_sources=processed_data.file_sources
        )
        
        # Create test split
        test_data = ProcessedData(
            texts=[processed_data.texts[i] for i in test_idx],
            labels=[processed_data.labels[i] for i in test_idx],
            metadata=[processed_data.metadata[i] for i in test_idx],
            file_sources=processed_data.file_sources
        )
        
        logger.info(f"Data split: {len(train_data.texts)} training, {len(test_data.texts)} test")
        return train_data, test_data
    
    def save_processed_data(self, processed_data: ProcessedData, save_path: str):
        """Save processed data to disk"""
        try:
            data_to_save = {
                'texts': processed_data.texts,
                'labels': processed_data.labels,
                'metadata': processed_data.metadata,
                'file_sources': processed_data.file_sources,
                'timestamp': np.datetime64('now')
            }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, default=str)
            
            logger.info(f"Processed data saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
    
    def load_processed_data(self, load_path: str) -> Optional[ProcessedData]:
        """Load processed data from disk"""
        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            processed_data = ProcessedData(
                texts=loaded_data['texts'],
                labels=loaded_data['labels'],
                metadata=loaded_data['metadata'],
                file_sources=loaded_data['file_sources']
            )
            
            logger.info(f"Loaded {len(processed_data.texts)} documents from {load_path}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            return None