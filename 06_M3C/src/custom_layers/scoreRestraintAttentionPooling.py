# Original architecture component; numerical operations and parameters are preserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreRestraintAttentionPooling(nn.Module):
    """Aggregate sequence hidden states using attention weights derived from lower-level predicted scores."""
    def __init__(self, num_scores: int, hidden_dim: int):
        """Initialize the score projection and pooling configuration.

num_scores is the number of concatenated lower-level scores.
hidden_dim documents the input hidden-state dimension."""
        super().__init__()
        # Linear layer implementing W and b in Equation 12.
        # Project the score vector to one salience logit.
        self.score_to_salience = nn.Linear(num_scores, 1)

    # Continuation of ScoreRestraintAttentionPooling.

    def forward(self, 
                hidden_states: torch.Tensor, 
                phone_scores: torch.Tensor,
                mdd_scores: torch.Tensor, 
                word_scores: torch.Tensor = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        """Pool hidden_states using phoneme scores and optional word scores.

The optional mask marks valid sequence positions with True.
Return one aggregated hidden-state vector per batch element."""
        
        # Step 1: prepare and concatenate scores.
        # Input to Equation 12: [p_i, w_i^0, w_i^1, w_i^2].
        # Concatenate score tensors along the final dimension.
        # Shape: + ->

        if word_scores is None:
            # When word scores are absent, use the available phoneme and MDD scores.
            combined_scores = torch.cat([phone_scores, mdd_scores], dim=-1)
        else:
            # Concatenate all available scores.
            combined_scores = torch.cat([phone_scores, mdd_scores, word_scores], dim=-1)
        
        # Step 2: compute salience (Equation 12).
        # s_i = GELU(W(...) + b)
        # Apply the linear layer and GELU activation.
        # Shape: ->
        salience_scores = self.score_to_salience(combined_scores)
        salience_scores = F.gelu(salience_scores)
        
        # Step 3: compute attention weights (Equation 13).
        # α_i = softmax(s_i)
        # Mask padded positions before softmax so that
        # padding tokens receive no attention.
        if mask is not None:
            # Add a dimension for broadcasting.
            mask = mask.unsqueeze(-1)
            # Assign a large negative value wherever the mask is False (padding)
            # so that those positions have approximately zero softmax probability.
            salience_scores = salience_scores.masked_fill(mask == 0, -1e9)
                

        # Apply softmax along the sequence dimension (T).
        # squeeze(-1) removes the final singleton dimension.
        # Shape: ->
        attention_weights = F.softmax(salience_scores, dim=0).squeeze(-1)
        
        # Step 4: weighted aggregation (Equation 14).
        # h_agg = (1/N) * Σ(α_l * h_utt^l)
        
        # Reshape the weights for elementwise multiplication
        # with hidden_states.
        # Shape: ->
        attention_weights_expanded = attention_weights.unsqueeze(-1)
        
        # Weight the hidden states elementwise.
        # Shape: * ->
        weighted_hidden_states = hidden_states * attention_weights_expanded
        
        # Sum along the sequence dimension.
        # Shape: ->
        aggregated_vector = torch.sum(weighted_hidden_states, dim=0)
        
        
        # Apply length normalization (1/N).
        # N counts non-padding positions in each sequence.
        if mask is not None:
            # Sum the mask to obtain each sequence length.
            # Shape: ->
            sequence_lengths = mask.sum(dim=0)
            # Add a dimension for division.
            sequence_lengths = sequence_lengths.unsqueeze(-1)
            # Avoid division by zero for empty sequences.
            sequence_lengths = torch.clamp(sequence_lengths, min=1)
        else:
            # Without a mask, all sequences share the same length.
            sequence_lengths = hidden_states.shape[0]

        # Final normalization.
        # Shape: / ->
        final_aggregated_vector = aggregated_vector / sequence_lengths
                
        return final_aggregated_vector