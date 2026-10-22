# n-cartpole ESP32-S3 firmware (ESP-IDF v5.3). Build: idf.py build (or `task fw-build`).
# The app version comes from version.txt (CI writes it from the fw-v* tag).
cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(n_cartpole_fw)
