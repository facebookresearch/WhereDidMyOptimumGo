# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json


class ConfigurationReader(object):

    def __init__(self, configuration_type="json"):
        """
        """

    def load(self, filename):
        with open(filename, 'r') as file:
            config = json.load(file)

        # TODO: recursive
        for k, v in config.items():
            if isinstance(v, str) and  ".json" in v:
                with open(v, 'r') as file:
                    file_contents = json.load(file)
                    config[k] = file_contents
        return config
