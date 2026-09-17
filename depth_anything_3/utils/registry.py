# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

try:
    from addict import Dict
except ImportError:
    class Dict(dict):
        @classmethod
        def __class_getitem__(cls, item):
            return cls
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)
        def __setattr__(self, name, value):
            self[name] = value
        def __delattr__(self, name):
            try:
                del self[name]
            except KeyError:
                raise AttributeError(name)


class Registry(Dict[str, Any]):
    def __init__(self):
        super().__init__()
        self._map = Dict({})

    def register(self, name=None):
        def decorator(cls):
            key = name or cls.__name__
            self._map[key] = cls
            return cls

        return decorator

    def get(self, name):
        return self._map[name]

    def all(self):
        return self._map
