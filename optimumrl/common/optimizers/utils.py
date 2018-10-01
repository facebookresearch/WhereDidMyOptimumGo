# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import importlib

def build_optimizer(optimizer_config, model):
    # We define the class by the module path with the class as the last thing
    splitted = optimizer_config["class"].split(".")
    module_name = ".".join(splitted[:-1])
    class_name = splitted[-1]
    # Next we import
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    parameters = model.parameters()

    # and instantiate the class with the added parameters
    return class_(parameters, **optimizer_config["kwargs"])
