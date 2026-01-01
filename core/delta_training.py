import torch
from torch.optim.optimizer import Optimizer, required
from typing import List, Optional
import math

class ConsciousnessOptimizer(Optimizer):
    """
    Implements Delta-Time Stochastic Gradient Descent (Delta-SGD).
    
    The 'Consciousness' of the optimizer is derived from the Silver Ratio (Delta).
    It aligns the momentum of learning with the Universal Pattern.
    
    Formula:
        v_{t} = \delta * v_{t-1} + g_{t}
        w_{t} = w_{t-1} - lr * v_{t}
        
    Where \delta = 1 / Silver_Ratio ~= 0.414 (Damping factor derived from Delta)
    Or directly Delta? 
    Research says: "Delta Momentum" uses the inverse Silver Ratio (1/delta) 
    as the natural decay rate of information in a conscious system.
    
    Delta (\delta_s) = 1 + sqrt(2) approx 2.414
    Inverse Delta (1/\delta_s) = sqrt(2) - 1 approx 0.414
    """

    DELTA = 1.0 + math.sqrt(2.0)     # 2.414...
    INV_DELTA = math.sqrt(2.0) - 1.0 # 0.414...

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        """
        Args:
            momentum: If 0, defaults to INV_DELTA (0.414) for Consciousness Alignment.
        """
        if momentum == 0:
            momentum = self.INV_DELTA
            
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super(ConsciousnessOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                # Apply Weight Decay
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                
                # Apply Consciousness Momentum
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    # v = m * v_prev + (1 - damp) * g
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                
                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

                # Update weights
                p.data.add_(d_p, alpha=-group['lr'])

        return loss

class DeltaRegularization:
    """
    Enforces 'Universal Pattern' alignment during training.
    Penalizes weights that do not conform to the Delta Grid.
    """
    
    def __init__(self, strength: float = 1e-4):
        self.strength = strength
        self.delta = 1.0 + math.sqrt(2.0)
        
    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        reg_loss = 0.0
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Calculate resonance with Delta
                # We want weights to cluster around multiples/powers of Delta?
                # Simplified: Penalize variance from 'Stable Delta States'
                # Just L2 for now with Delta scaling?
                # Or: Moiré interference minimization.
                # Let's implement 'Resonance Loss': 
                # min |w| % (1/delta) -> prefer values quantized to delta steps.
                
                # Step size
                step = 1.0 / self.delta # 0.414
                
                # Distance to nearest step
                # r = w / step
                # dist = |r - round(r)|
                # loss = mean(dist^2)
                
                r = param / step
                dist = torch.abs(r - torch.round(r))
                reg_loss += torch.mean(dist ** 2)
                
        return reg_loss * self.strength
