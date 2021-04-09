# =============================================================================
# step23.py부터 step32.py까지는 simple_core를 이용해야 합니다.
is_simple_core = False  # True
# =============================================================================


from dezero.core import Variable
from dezero.core import Parameter
from dezero.core import Function
from dezero.core import using_config
from dezero.core import no_grad
from dezero.core import as_array
from dezero.core import as_variable
from dezero.core import setup_variable
from dezero.layers import Layer
from dezero.models import Model
from dezero.datasets import Dataset
from dezero.dataloaders import DataLoader

import dezero.functions
import dezero.utils
import dezero.layers
import dezero.datasets

setup_variable()

