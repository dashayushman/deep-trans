# Description:
# Example neural translation models.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "package",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":data_utils",
        ":seq2seq_model",
    ],
)

py_library(
    name = "data_utils",
    srcs = [
        "data_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_library(
    name = "seq2seq_model",
    srcs = [
        "seq2seq_model.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":data_utils",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "transliterate",
    srcs = [
        "transliterate.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":data_utils",
        ":seq2seq_model",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "transliterate_test",
    size = "medium",
    srcs = [
        "transliterate.py",
    ],
    args = [
        "--self_test=True",
    ],
    main = "transliterate.py",
    srcs_version = "PY2AND3",
    deps = [
        ":data_utils",
        ":seq2seq_model",
        "//tensorflow:tensorflow_py",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
