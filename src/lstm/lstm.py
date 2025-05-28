import numpy as np
import h5py
from typing import Tuple, Optional, Dict, Any, List

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def softmax(x: np.ndarray) -> np.ndarray:
    shifted_x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class EmbeddingLayer:    
    def __init__(self, embedding_weights: np.ndarray):
        self.embedding_weights = embedding_weights
        self.vocab_size, self.embedding_dim = embedding_weights.shape
    
    def forward(self, token_indices: np.ndarray) -> np.ndarray:
        return self.embedding_weights[token_indices]

class LSTMLayer:
    def __init__(self, lstm_weights: Dict[str, np.ndarray]):
        self.input_weights_input = lstm_weights['input_weights_input']
        self.input_weights_forget = lstm_weights['input_weights_forget']
        self.input_weights_candidate = lstm_weights['input_weights_candidate']
        self.input_weights_output = lstm_weights['input_weights_output']
        
        self.hidden_weights_input = lstm_weights['hidden_weights_input']
        self.hidden_weights_forget = lstm_weights['hidden_weights_forget']
        self.hidden_weights_candidate = lstm_weights['hidden_weights_candidate']
        self.hidden_weights_output = lstm_weights['hidden_weights_output']
        
        self.bias_input = lstm_weights['bias_input']
        self.bias_forget = lstm_weights['bias_forget']
        self.bias_candidate = lstm_weights['bias_candidate']
        self.bias_output = lstm_weights['bias_output']

        self.hidden_units = self.input_weights_forget.shape[1]
        self.input_dim = self.input_weights_forget.shape[0]
    
    def forward(self, 
                input_sequence: np.ndarray, 
                mask: Optional[np.ndarray] = None,
                initial_hidden_state: Optional[np.ndarray] = None,
                initial_cell_state: Optional[np.ndarray] = None,
                return_sequences: bool = True) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        batch_size, sequence_length, _ = input_sequence.shape

        if mask is None:
            mask = np.ones((batch_size, sequence_length), dtype=bool)

        if initial_hidden_state is None:
            hidden_state = np.zeros((batch_size, self.hidden_units))
        else:
            hidden_state = initial_hidden_state.copy()
            
        if initial_cell_state is None:
            cell_state = np.zeros((batch_size, self.hidden_units))
        else:
            cell_state = initial_cell_state.copy()
     
        if return_sequences:
            all_hidden_states = np.zeros((batch_size, sequence_length, self.hidden_units))
        
        for time_step in range(sequence_length):
            current_input = input_sequence[:, time_step, :]  
            mask_t = mask[:, time_step]

            h_prev = hidden_state.copy()
            c_prev = cell_state.copy()

            input_gate = sigmoid(
                current_input @ self.input_weights_input + 
                hidden_state @ self.hidden_weights_input + 
                self.bias_input
            )

            forget_gate = sigmoid(
                current_input @ self.input_weights_forget + 
                hidden_state @ self.hidden_weights_forget + 
                self.bias_forget
            )
            
            candidate_values = tanh(
                current_input @ self.input_weights_candidate + 
                hidden_state @ self.hidden_weights_candidate + 
                self.bias_candidate
            )
            
            output_gate = sigmoid(
                current_input @ self.input_weights_output + 
                hidden_state @ self.hidden_weights_output + 
                self.bias_output
            )
            
            cell_state = forget_gate * cell_state + input_gate * candidate_values
            
            new_c = cell_state
            new_h = output_gate * tanh(new_c)
            
            hidden_state = np.where(mask_t[:, None], new_h, h_prev)
            cell_state   = np.where(mask_t[:, None], new_c, c_prev)

            if return_sequences:
                all_hidden_states[:, time_step, :] = hidden_state
        
        if return_sequences:
            return all_hidden_states, (hidden_state, cell_state)
        else:
            return hidden_state, (hidden_state, cell_state)

class BidirectionalLSTMLayer:
    def __init__(self,
                 fw_weights: Dict[str, np.ndarray],
                 bw_weights: Dict[str, np.ndarray]):

        self.fw = LSTMLayer(fw_weights)
        self.bw = LSTMLayer(bw_weights)
        self.hidden_units = self.fw.hidden_units

    def forward(self,
                input_sequence: np.ndarray,
                mask: Optional[np.ndarray] = None,
                initial_hidden_state=None,
                initial_cell_state=None,
                return_sequences: bool = False
               ):
        out_fw, (h_fw, c_fw) = self.fw.forward(
            input_sequence, mask,
            initial_hidden_state, initial_cell_state,
            return_sequences=return_sequences
        )
        rev_x   = input_sequence[:, ::-1, :]
        rev_mask= mask[:, ::-1] if mask is not None else None
        out_bw, (h_bw, c_bw) = self.bw.forward(
            rev_x, rev_mask,
            initial_hidden_state, initial_cell_state,
            return_sequences=return_sequences
        )

        if return_sequences:
            out_bw = out_bw[:, ::-1, :]
            return np.concatenate([out_fw, out_bw], axis=-1), (None, None)
        else:
            h = np.concatenate([h_fw, h_bw], axis=-1)
            return h, (h, None)

class DropoutLayer:
    def __init__(self, dropout_rate: float):
        self.dropout_rate = dropout_rate
    
    def forward(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        if not training or self.dropout_rate == 0.0:
            return inputs

        keep_probability = 1.0 - self.dropout_rate
        dropout_mask = (np.random.rand(*inputs.shape) < keep_probability)
  
        return inputs * dropout_mask / keep_probability

class DenseLayer:
    def __init__(self, 
                 weights: np.ndarray, 
                 bias: np.ndarray, 
                 activation: Optional[str] = None):
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.input_dim, self.output_dim = weights.shape
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        linear_output = inputs @ self.weights + self.bias
        
        if self.activation == 'softmax':
            return softmax(linear_output)
        elif self.activation == 'relu':
            return np.maximum(0, linear_output)
        elif self.activation == 'tanh':
            return tanh(linear_output)
        elif self.activation == 'sigmoid':
            return sigmoid(linear_output)
        else:
            return linear_output

class LSTMModel:
    def __init__(self, layers: List[Any]):
        self.layers = layers
        rec_types = (LSTMLayer, BidirectionalLSTMLayer)
        rec_idxs = [i for i, L in enumerate(layers) if isinstance(L, rec_types)]
        if not rec_idxs:
            raise ValueError("No recurrent layers found!")
        self._last_lstm_idx = max(rec_idxs)

    def forward(self, token_indices: np.ndarray, training: bool = False) -> np.ndarray:
        mask = (token_indices != 0)
        current = token_indices
        rec_types = (LSTMLayer, BidirectionalLSTMLayer)

        for idx, layer in enumerate(self.layers):
            if isinstance(layer, rec_types):
                return_seq = (idx != self._last_lstm_idx)
                current, _ = layer.forward(current,
                                          mask=mask,
                                          return_sequences=return_seq)
            elif isinstance(layer, DropoutLayer):
                current = layer.forward(current, training=training)
            else:
                current = layer.forward(current)
        return current

    def predict(self, token_indices: np.ndarray) -> np.ndarray:
        return self.forward(token_indices, training=False)

def load_lstm_weights_from_keras(
    h5_file_path: str,
    bidirectional: bool = False,
    num_lstm_layers: int = 1
) -> Dict[str, Any]:
    weights: Dict[str, Any] = {}

    with h5py.File(h5_file_path, 'r') as h5:
        weights['embedding'] = h5['model_weights']\
                                ['embedding']['embedding']\
                                ['embeddings'][()]

        for layer_idx in range(num_lstm_layers):
            if not bidirectional:
                keras_layer_name = f'lstm_{layer_idx}' if layer_idx > 0 else 'lstm'
                grp = h5['model_weights'][keras_layer_name][keras_layer_name]
                cell_grp = grp['lstm_cell']

                K = cell_grp['kernel'][()]
                R = cell_grp['recurrent_kernel'][()]
                B = cell_grp['bias'][()]

                hidden = R.shape[0]
                Wi, Wf, Wg, Wo = np.split(K, 4, axis=1)
                Ui, Uf, Ug, Uo = np.split(R, 4, axis=1)
                bi, bf, bg, bo = np.split(B, 4)

                weights[f'lstm_{layer_idx}'] = {
                    'input_weights_input':  Wi,
                    'input_weights_forget': Wf,
                    'input_weights_candidate': Wg,
                    'input_weights_output': Wo,
                    'hidden_weights_input':  Ui,
                    'hidden_weights_forget': Uf,
                    'hidden_weights_candidate': Ug,
                    'hidden_weights_output': Uo,
                    'bias_input':  bi,
                    'bias_forget': bf,
                    'bias_candidate': bg,
                    'bias_output': bo,
                }
            else:
                bi_name = f'bidirectional_{layer_idx}' if layer_idx>0 else 'bidirectional'
                bidi_grp = h5['model_weights'][bi_name][bi_name]

                fwd_grp = bidi_grp['forward_lstm']['lstm_cell']
                bwd_grp = bidi_grp['backward_lstm']['lstm_cell']

                def _extract(cell_grp):
                    K = cell_grp['kernel'][()]
                    R = cell_grp['recurrent_kernel'][()]
                    B = cell_grp['bias'][()]
                    hidden = R.shape[0]
                    Wi, Wf, Wg, Wo = np.split(K, 4, axis=1)
                    Ui, Uf, Ug, Uo = np.split(R, 4, axis=1)
                    bi, bf, bg, bo = np.split(B, 4)
                    return {
                        'input_weights_input':  Wi,
                        'input_weights_forget': Wf,
                        'input_weights_candidate': Wg,
                        'input_weights_output': Wo,
                        'hidden_weights_input':  Ui,
                        'hidden_weights_forget': Uf,
                        'hidden_weights_candidate': Ug,
                        'hidden_weights_output': Uo,
                        'bias_input':  bi,
                        'bias_forget': bf,
                        'bias_candidate': bg,
                        'bias_output': bo,
                    }

                weights[f'lstm_fw_{layer_idx}'] = _extract(fwd_grp)
                weights[f'lstm_bw_{layer_idx}'] = _extract(bwd_grp)

        dense_grp = h5['model_weights']['dense']['dense']
        W = dense_grp['kernel'][()]
        b = dense_grp['bias'][()]
        weights['dense'] = (W, b)

    return weights


def create_lstm_model_from_weights(
    weights_dict: Dict[str, Any],
    dropout_rate: float = 0.5,
    bidirectional: bool = False
) -> LSTMModel:
    layers: List[Any] = []
    layers.append(EmbeddingLayer(weights_dict['embedding']))

    i = 0
    while True:
        if not bidirectional:
            key = f'lstm_{i}'
            if key not in weights_dict:
                break
            layers.append(LSTMLayer(weights_dict[key]))
        else:
            fwd = f'lstm_fw_{i}'
            bwd = f'lstm_bw_{i}'
            if fwd not in weights_dict or bwd not in weights_dict:
                break
            layers.append(
              BidirectionalLSTMLayer(
                weights_dict[fwd],
                weights_dict[bwd]
              )
            )
        i += 1

    layers.append(DropoutLayer(dropout_rate))
    W, b = weights_dict['dense']
    layers.append(DenseLayer(W, b, activation='softmax'))

    return LSTMModel(layers)

def test_lstm_model(model: LSTMModel, 
                   test_sequences: np.ndarray, 
                   test_labels: np.ndarray) -> Tuple[float, np.ndarray]:
    predictions_proba = model.predict(test_sequences)
    predictions = np.argmax(predictions_proba, axis=1)
    
    accuracy = np.mean(predictions == test_labels)
    
    return accuracy, predictions

def compute_macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    f1_scores = []
    for class_label in classes:
        tp = np.sum((y_true == class_label) & (y_pred == class_label))
        fp = np.sum((y_true != class_label) & (y_pred == class_label))
        fn = np.sum((y_true == class_label) & (y_pred != class_label))
   
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
 
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    
    return np.mean(f1_scores)