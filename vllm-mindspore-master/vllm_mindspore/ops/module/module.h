#ifndef VLLM_MINDSPORE_OPS_MODULE_MODULE_H
#define VLLM_MINDSPORE_OPS_MODULE_MODULE_H

#include <pybind11/pybind11.h>
#include <functional>
#include <vector>
#include <string>

// Define the type of module registration functions
using ModuleRegisterFunction = std::function<void(pybind11::module_ &)>;

// Module registry class
class ModuleRegistry {
 public:
  // Get the singleton instance
  static ModuleRegistry &Instance() {
    static ModuleRegistry instance;
    return instance;
  }

  // Register a module function
  void Register(const ModuleRegisterFunction &func) { functions_.push_back(func); }

  // Call all registered module functions
  void RegisterAll(pybind11::module_ &m) {
    for (const auto &func : functions_) {
      func(m);
    }
  }

 private:
  ModuleRegistry() = default;
  ~ModuleRegistry() = default;

  // Disable copy and assignment
  ModuleRegistry(const ModuleRegistry &) = delete;
  ModuleRegistry &operator=(const ModuleRegistry &) = delete;

  // Store all registered functions
  std::vector<ModuleRegisterFunction> functions_;
};

// Define a macro to register module functions
#define MS_EXTENSION_MODULE(func)                                                \
  static void func##_register(pybind11::module_ &);                              \
  namespace {                                                                    \
  struct func##_registrar {                                                      \
    func##_registrar() { ModuleRegistry::Instance().Register(func##_register); } \
  };                                                                             \
  static func##_registrar registrar_instance;                                    \
  }                                                                              \
  static void func##_register(pybind11::module_ &m)

#endif  // VLLM_MINDSPORE_OPS_MODULE_MODULE_H
