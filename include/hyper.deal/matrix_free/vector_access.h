// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the hyper.deal authors
//
// This file is part of the hyper.deal library.
//
// The hyper.deal library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.MD at
// the top level directory of hyper.deal.
//
// ---------------------------------------------------------------------

#ifndef DEALII_LINEARALGEBRA_SHAREDMPI_EVALUATION
#define DEALII_LINEARALGEBRA_SHAREDMPI_EVALUATION

#include <hyper.deal/base/config.h>

#include <hyper.deal/matrix_free/dof_info.h>
#include <hyper.deal/matrix_free/face_info.h>
#include <hyper.deal/matrix_free/shape_info.h>
#include <hyper.deal/matrix_free/vector_access_kernels.h>

namespace hyperdeal
{
  namespace internal
  {
    namespace MatrixFreeFunctions
    {
      /**
       * Read/write functions translating macro IDs to addresses.
       *
       * TODO: move to deal.II or make it similar to VectorReader/VectorWriter.
       */
      template <typename Number>
      class ReadWriteOperation
      {
      public:
        ReadWriteOperation(
          const hyperdeal::internal::MatrixFreeFunctions::DoFInfo & dof_info,
          const hyperdeal::internal::MatrixFreeFunctions::FaceInfo &face_info,
          const hyperdeal::internal::MatrixFreeFunctions::ShapeInfo<Number>
            &shape_info);

        template <int dim,
                  int degree,
                  typename VectorOperation,
                  typename VectorizedArrayType>
        void
        process_cell(const VectorOperation &      operation,
                     const std::vector<double *> &data_others,
                     VectorizedArrayType *        dst,
                     const unsigned int           cell_batch_number) const;


        template <int dim,
                  int degree,
                  typename VectorOperation,
                  typename VectorizedArrayType>
        void
        process_face(const VectorOperation &      operation,
                     const std::vector<double *> &data_others,
                     VectorizedArrayType *        dst,
                     const unsigned int           face_batch_number,
                     const unsigned int           face_no,
                     const unsigned int           side) const;


        template <int dim, int degree, typename VectorizedArrayType>
        void
        distribute_local_to_global_face_batched(
          std::vector<double *> &data_others,
          const Number *         src,
          const unsigned int     face_batch_number,
          const unsigned int     face_no,
          const unsigned int     side) const;


      private:
        const std::array<std::vector<unsigned char>, 4>
          &n_vectorization_lanes_filled;
        const std::array<std::vector<std::pair<unsigned int, unsigned int>>, 4>
          &                                     dof_indices_contiguous_ptr;
        const std::array<std::vector<bool>, 4> &face_type;
        const std::array<std::vector<bool>, 4> &face_all;

        const std::vector<std::vector<unsigned int>> &face_to_cell_index_nodal;
      };



      template <typename Number>
      ReadWriteOperation<Number>::ReadWriteOperation(
        const hyperdeal::internal::MatrixFreeFunctions::DoFInfo & dof_info,
        const hyperdeal::internal::MatrixFreeFunctions::FaceInfo &face_info,
        const hyperdeal::internal::MatrixFreeFunctions::ShapeInfo<Number>
          &shape_info)
        : n_vectorization_lanes_filled(dof_info.n_vectorization_lanes_filled)
        , dof_indices_contiguous_ptr(dof_info.dof_indices_contiguous_ptr)
        , face_type(face_info.face_type)
        , face_all(face_info.face_all)
        , face_to_cell_index_nodal(shape_info.face_to_cell_index_nodal)
      {}



      template <typename Number>
      template <int dim,
                int degree,
                typename VectorOperation,
                typename VectorizedArrayType>
      void
      ReadWriteOperation<Number>::process_cell(
        const VectorOperation &      operation,
        const std::vector<double *> &global,
        VectorizedArrayType *        local,
        const unsigned int           cell_batch_number) const
      {
        static const unsigned int v_len = VectorizedArrayType::size();
        static const unsigned int n_dofs_per_cell =
          dealii::Utilities::pow<unsigned int>(degree + 1, dim);

        // step 1: get pointer to the first dof of the cell in the sm-domain
        std::array<Number *, v_len> global_ptr;
        for (unsigned int v = 0;
             v < n_vectorization_lanes_filled[2][cell_batch_number] &&
             v < v_len;
             v++)
          {
            const auto sm_ptr =
              dof_indices_contiguous_ptr[2][v_len * cell_batch_number + v];
            global_ptr[v] = global[sm_ptr.first] + sm_ptr.second;
          }

        // step 2: process dofs
        if (n_vectorization_lanes_filled[2][cell_batch_number] == v_len)
          // case 1: all lanes are filled -> use optimized function
          operation.process_dofs_vectorized_transpose(n_dofs_per_cell,
                                                      global_ptr,
                                                      local);
        else
          // case 2: some lanes are empty
          for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
            for (unsigned int v = 0;
                 v < n_vectorization_lanes_filled[2][cell_batch_number] &&
                 v < v_len;
                 v++)
              operation.process_dof(global_ptr[v][i], local[i][v]);
      }



      template <typename Number>
      template <int dim,
                int degree,
                typename VectorOperation,
                typename VectorizedArrayType>
      void
      ReadWriteOperation<Number>::process_face(
        const VectorOperation &      operation,
        const std::vector<double *> &global,
        VectorizedArrayType *        local,
        const unsigned int           face_batch_number,
        const unsigned int           face_no,
        const unsigned int           side) const
      {
        static const unsigned int v_len = VectorizedArrayType::size();
        static const unsigned int n_dofs_per_face =
          dealii::Utilities::pow<unsigned int>(degree + 1, dim - 1);

        std::array<Number *, v_len> srcs;
        for (unsigned int v = 0;
             v < n_vectorization_lanes_filled[side][face_batch_number] &&
             v < v_len;
             v++)
          {
            const auto sm_ptr =
              dof_indices_contiguous_ptr[side][v_len * face_batch_number + v];
            srcs[v] = global[sm_ptr.first] + sm_ptr.second;
          }

        if (n_vectorization_lanes_filled[side][face_batch_number] == v_len &&
            face_all[side][face_batch_number])
          {
            if (face_type[side][v_len * face_batch_number])
              {
                // case 1: read from buffers
                for (unsigned int i = 0; i < n_dofs_per_face; ++i)
                  for (unsigned int v = 0; v < v_len; ++v)
                    operation.process_dof(srcs[v][i], local[i][v]);
              }
            else
              {
                // case 2: read from shared memory
                for (unsigned int i = 0; i < n_dofs_per_face; ++i)
                  for (unsigned int v = 0; v < v_len; ++v)
                    operation.process_dof(
                      srcs[v][face_to_cell_index_nodal[face_no][i]],
                      local[i][v]);
              }
          }
        else
          for (unsigned int v = 0;
               v < n_vectorization_lanes_filled[side][face_batch_number] &&
               v < v_len;
               v++)
            {
              if (face_type[side][v_len * face_batch_number + v])
                {
                  // case 1: read from buffers
                  for (unsigned int i = 0; i < n_dofs_per_face; ++i)
                    operation.process_dof(srcs[v][i], local[i][v]);
                }
              else
                {
                  // case 2: read from shared memory
                  for (unsigned int i = 0; i < n_dofs_per_face; ++i)
                    operation.process_dof(
                      srcs[v][face_to_cell_index_nodal[face_no][i]],
                      local[i][v]);
                }
            }
      }



      template <typename Number>
      template <int dim, int degree, typename VectorizedArrayType>
      void
      ReadWriteOperation<Number>::distribute_local_to_global_face_batched(
        std::vector<double *> &global,
        const Number *         local,
        const unsigned int     face_batch_number,
        const unsigned int     face_no,
        const unsigned int     side) const
      {
        using DA =
          VectorReaderWriterKernels<dim, degree, Number, VectorizedArrayType>;
        static const int v_len = VectorizedArrayType::size();

        std::array<Number *, v_len> dsts;
        for (unsigned int v = 0;
             v < n_vectorization_lanes_filled[side][face_batch_number] &&
             v < v_len;
             v++)
          {
            const auto sm_ptr =
              dof_indices_contiguous_ptr[side][v_len * face_batch_number + v];
            dsts[v] = global[sm_ptr.first] + sm_ptr.second;
          }

        if (n_vectorization_lanes_filled[side][face_batch_number] == v_len &&
            face_all[side][face_batch_number])
          DA::template scatterv_face<v_len, true>(
            dsts, face_no, local, face_type[side][v_len * face_batch_number]);
        else
          for (unsigned int v = 0;
               v < n_vectorization_lanes_filled[side][face_batch_number] &&
               v < v_len;
               v++)
            DA::template scatter_face<v_len, true>(
              dsts[v],
              face_no,
              local + v,
              face_type[side][v_len * face_batch_number + v]);
      }



      template <typename Number, typename VectorizedArrayType>
      struct VectorReader
      {
        void
        process_dof(const Number &global, Number &local) const
        {
          local = global;
        }

        void
        process_dofs_vectorized_transpose(
          const unsigned int dofs_per_cell,
          const std::array<Number *, VectorizedArrayType::size()> &global_ptr,
          VectorizedArrayType *                                    local) const
        {
          vectorized_load_and_transpose(dofs_per_cell, global_ptr, local);
        }
      };



      template <typename Number, typename VectorizedArrayType>
      struct VectorSetter
      {
        void
        process_dof(Number &global, const Number &local) const
        {
          global = local;
        }

        void
        process_dofs_vectorized_transpose(
          const unsigned int                                 dofs_per_cell,
          std::array<Number *, VectorizedArrayType::size()> &global_ptr,
          const VectorizedArrayType *                        local) const
        {
          vectorized_transpose_and_store(false,
                                         dofs_per_cell,
                                         local,
                                         global_ptr);
        }
      };

    } // namespace MatrixFreeFunctions
  }   // namespace internal
} // namespace hyperdeal

#endif
