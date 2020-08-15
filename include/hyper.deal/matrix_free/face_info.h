#ifndef HYPERDEAL_NDIM_MATRIXFREE_FACE_INFO
#define HYPERDEAL_NDIM_MATRIXFREE_FACE_INFO

#include <hyper.deal/base/config.h>

namespace hyperdeal
{
  namespace internal
  {
    namespace MatrixFreeFunctions
    {
      /**
       * A data structure that holds the connectivity between the faces and the
       * cells.
       */
      struct FaceInfo
      {
        /**
         * Caches the face number of a macro face.
         *
         * @note (0) FCL (interior); (1) FCL (exterior); (2) TODO; (3) TODO
         */
        std::array<std::vector<unsigned int>, 4> no_faces;

        /**
         * Caches the face orientation of a macro face.
         *
         * @note TODO
         */
        std::array<std::vector<unsigned int>, 4> face_orientations;

        /**
         * Stores for each face the type.Type
         * means in this context if their dofs are stored in faces or
         * are embedded in a cell. Example: ghost faces vs. faces of
         * cells owned by a process in the same shared memory domain.
         *
         * @note (0) FCL (interior); (1) FCL (exterior);
         *       (2) empty (TODO - remove); (3) ECL (exterior)
         */
        std::array<std::vector<bool>, 4> face_type;

        /**
         * Caches if all faces have the same type, face number, and face
         * orientation when vectorizing.
         *
         * @note (0) FCL (interior); (1) FCL (exterior);
         *       (2) empty (TODO - remove); (3) ECL (exterior)
         */
        std::array<std::vector<bool>, 4> face_all;


        std::size_t
        memory_consumption() const
        {
          return dealii::MemoryConsumption::memory_consumption(no_faces) +
                 dealii::MemoryConsumption::memory_consumption(face_all) +
                 dealii::MemoryConsumption::memory_consumption(face_type);
        }
      };
    } // namespace MatrixFreeFunctions
  }   // namespace internal
} // namespace hyperdeal

#endif
