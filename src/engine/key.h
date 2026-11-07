#pragma once

#include <arrow/api.h>

#include <string>

namespace qe {

// Serializes one value from a typed array into a byte string suitable for
// hashing/equality. The encoding is type-tagged and length-prefixed, so distinct
// values cannot collide by concatenation. Reading directly from the array avoids
// allocating an Arrow scalar for every row.
arrow::Status AppendArrayKey(const arrow::Array& array, int64_t index, std::string& out);

bool IsSupportedKeyType(const arrow::DataType& type);

// True when two key types can be compared after casting to a common type.
bool AreKeyTypesCompatible(const arrow::DataType& left, const arrow::DataType& right);

arrow::Result<std::shared_ptr<arrow::DataType>> CommonKeyType(const arrow::DataType& left,
                                                              const arrow::DataType& right);

} // namespace qe
