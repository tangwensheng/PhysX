// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.

#include "PxgCudaMemoryAllocator.h"
#include "PxgPBDParticleSystemCore.h"
#include "PxgParticleSystem.h"
#include "foundation/PxAssert.h"
#include "common/PxProfileZone.h"
#include "cudamanager/PxCudaContextManager.h"
#include "PxgCudaSolverCore.h"
#include "PxgCudaBroadPhaseSap.h"
#include "PxgSimulationController.h"
#include "PxgSimulationCore.h"
#include "PxgNarrowphaseCore.h"
#include "PxgNphaseImplementationContext.h"
#include "PxgKernelWrangler.h"
#include "CudaKernelWrangler.h"
#include "PxgKernelIndices.h"
#include "PxgRadixSortDesc.h"
#include "PxgParticleSystemCoreKernelIndices.h"
#include "PxgRadixSortKernelIndices.h"
#include "PxgRadixSortCore.h"
#include "PxgCudaUtils.h"
#include "DyParticleSystem.h"
#include "PxgContext.h"
#include "CmRandom.h"

#include "cudamanager/PxCudaContext.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define PS_GPU_DEBUG 0
#define PS_GPU_SPARSE_GRID_DEBUG 0
#define PS_SUBGRIDS_DEBUG 0
#define PS_SUBGRIDS_VERBOSE_DEBUG 0


namespace physx
{
	#if defined(PX_DCU_PORT) && PX_DCU_PORT
	void dcuPbdPipelineCheckpoint(PxCudaContext* cudaContext, CUstream stream, const char* phase,
		CUdeviceptr deviceCount, PxU32 hostValue)
	{
		const char* enabled = std::getenv("PX_DCU_PBD_PIPELINE_DIAG");
		if (!enabled || enabled[0] != '1')
			return;

		static std::atomic<unsigned long long> sequence(0);
		const unsigned long long checkpoint = sequence.fetch_add(1, std::memory_order_relaxed) + 1;
		const PxCUresult syncResult = cudaContext->streamSynchronize(stream);
		PxU32 count = 0;
		PxCUresult copyResult = CUDA_SUCCESS;
		if (syncResult == CUDA_SUCCESS && deviceCount)
			copyResult = cudaContext->memcpyDtoH(&count, deviceCount, sizeof(count));

		std::printf("[DCU PBD PIPELINE] marker=PX_DCU_PBD_PIPELINE_DIAG_V1 seq=%llu phase=%s sync=%d copy=%d device_count=%u host_value=%u\n",
			checkpoint, phase, int(syncResult), int(copyResult), count, hostValue);
		std::fflush(stdout);
	}

	void dcuPbdPrimitiveSortCheckpoint(PxCudaContext* cudaContext, CUstream stream, const char* phase,
		CUdeviceptr deviceCount, PxU32 hostValue)
	{
		const char* enabled = std::getenv("PX_DCU_PBD_PRIMITIVE_PIPELINE_DIAG");
		if (!enabled || std::strcmp(enabled, "1") != 0)
			return;

		static std::atomic<unsigned long long> sequence(0);
		const unsigned long long checkpoint = sequence.fetch_add(1, std::memory_order_relaxed) + 1;
		const PxCUresult syncResult = cudaContext->streamSynchronize(stream);
		PxU32 count = 0;
		PxCUresult copyResult = CUDA_SUCCESS;
		if (syncResult == CUDA_SUCCESS && deviceCount)
			copyResult = cudaContext->memcpyDtoH(&count, deviceCount, sizeof(count));

		std::printf("[DCU PBD PRIMITIVE PIPELINE] marker=PX_DCU_PBD_PRIMITIVE_PIPELINE_DIAG_V2 source=sort seq=%llu phase=%s sync=%d copy=%d device_count=%u host_value=%u\n",
			checkpoint, phase, int(syncResult), int(copyResult), count, hostValue);
		std::fflush(stdout);
	}

	bool dcuPbdUseChunkedPrimitiveRadix()
	{
		static const bool enabled = []()
		{
			const char* value = std::getenv("PX_DCU_PBD_PRIMITIVE_RADIX_CHUNKED");
			return value && std::strcmp(value, "1") == 0;
		}();
		return enabled;
	}

	bool dcuPbdUseInlinePrimitiveRadix()
	{
		static const bool enabled = []()
		{
			const char* value = std::getenv("PX_DCU_PBD_PRIMITIVE_RADIX_INLINE");
			return value && std::strcmp(value, "1") == 0;
		}();
		return enabled;
	}

	bool dcuPbdUseFused64PrimitiveRadix()
	{
		static const bool enabled = []()
		{
			const char* value = std::getenv("PX_DCU_PBD_PRIMITIVE_RADIX_FUSED64");
			return value && std::strcmp(value, "1") == 0;
		}();
		return enabled;
	}

	bool dcuPbdPrimitiveSortDiagEnabled()
	{
		static const bool enabled = []()
		{
			const char* value = std::getenv("PX_DCU_PBD_PRIMITIVE_RANK_DIAG");
			return value && std::strcmp(value, "1") == 0;
		}();
		return enabled;
	}

	bool dcuPbdUseFusedParticleRanges()
	{
		static const bool enabled = []()
		{
			const char* value = std::getenv("PX_DCU_PBD_PARTICLE_RANGE_FUSED");
			return value && std::strcmp(value, "1") == 0;
		}();
		return enabled;
	}

	bool dcuPbdUseChunkedParticleRanges()
	{
		static const bool enabled = []()
		{
			const char* value = std::getenv("PX_DCU_PBD_PARTICLE_RANGE_CHUNKED");
			return value && std::strcmp(value, "1") == 0;
		}();
		return enabled;
	}

	bool dcuPbdUseParallelParticleRanges()
	{
		static const bool enabled = []()
		{
			const char* value = std::getenv("PX_DCU_PBD_PARTICLE_RANGE_PARALLEL");
			return value && std::strcmp(value, "1") == 0;
		}();
		return enabled;
	}

	bool dcuPbdParticleRangeDiagEnabled()
	{
		static const bool enabled = []()
		{
			const char* value = std::getenv("PX_DCU_PBD_PARTICLE_RANGE_DIAG");
			return value && std::strcmp(value, "1") == 0;
		}();
		return enabled;
	}

	void dcuPbdPrintParticleRangeModeOnce(bool fused, bool chunked, bool parallel)
	{
		static std::atomic<bool> printed(false);
		if (!printed.exchange(true, std::memory_order_relaxed))
		{
			std::printf("[DCU PBD RANGE] host_marker=%s mode=%s device_marker=%s scans_per_frame=%u first_launch_grid=%ux%ux%u second_launch_grid=%ux%ux%u second_launch_block=%ux%ux%u\n",
				parallel ? "PX_DCU_PBD_PARTICLE_RANGE_HOST_V62_PARALLEL_OPT_IN" : (chunked ? "PX_DCU_PBD_PARTICLE_RANGE_HOST_V61_CHUNKED_OPT_IN" : "PX_DCU_PBD_PARTICLE_RANGE_HOST_V60_OPT_IN"),
				parallel ? "parallel_v62" : (chunked ? "chunked_v61" : (fused ? "fused_v60" : "two_scan_v59")),
				parallel ? "PX_DCU_PBD_PARTICLE_RANGE_V62_PARALLEL_BOUNDARY_32" : (chunked ? "PX_DCU_PBD_PARTICLE_RANGE_V61_CHUNKED_SINGLE_SCAN" : (fused ? "PX_DCU_PBD_PARTICLE_RANGE_V60_FUSED_SINGLE_SCAN" : "PX_DCU_PARTICLE_CONTACT_RANGE_V59_DETERMINISTIC_LARGE")),
				fused ? 1u : 2u, fused ? 0u : 32u, 1u, 1u,
				fused ? 1u : 32u, 1u, 1u, (parallel || chunked) ? 32u : (fused ? 1u : 256u), 1u, 1u);
			std::fflush(stdout);
		}
	}

	bool dcuPbdValidateParticleRanges(PxCudaContext* cudaContext, CUstream stream,
		CUdeviceptr sortedContactsd, CUdeviceptr contactCountd, CUdeviceptr pairCountd,
		CUdeviceptr rangeStartd, CUdeviceptr rangeEndd, PxU32 maxContacts)
	{
		if (!dcuPbdParticleRangeDiagEnabled())
			return true;

		const PxCUresult syncResult = cudaContext->streamSynchronize(stream);
		PxU32 contactCount = 0;
		PxU32 pairCount = 0;
		PxCUresult contactCountResult = syncResult == CUDA_SUCCESS
			? cudaContext->memcpyDtoH(&contactCount, contactCountd, sizeof(contactCount)) : syncResult;
		PxCUresult pairCountResult = contactCountResult == CUDA_SUCCESS
			? cudaContext->memcpyDtoH(&pairCount, pairCountd, sizeof(pairCount)) : contactCountResult;

		if (syncResult != CUDA_SUCCESS || contactCountResult != CUDA_SUCCESS || pairCountResult != CUDA_SUCCESS ||
			contactCount > maxContacts || pairCount > contactCount)
		{
			std::printf("[DCU PBD RANGE DIAG] marker=PX_DCU_PBD_PARTICLE_RANGE_DIAG_V60 verdict=READBACK_OR_COUNT_FAILED sync=%d contact_copy=%d pair_copy=%d contacts=%u pairs=%u max_contacts=%u\n",
				int(syncResult), int(contactCountResult), int(pairCountResult), contactCount, pairCount, maxContacts);
			std::fflush(stdout);
			return false;
		}

		std::vector<PxgParticlePrimitiveContact> contacts(contactCount);
		std::vector<PxU32> rangeStarts(pairCount);
		std::vector<PxU32> rangeEnds(pairCount);
		PxCUresult contactsResult = CUDA_SUCCESS;
		PxCUresult startsResult = CUDA_SUCCESS;
		PxCUresult endsResult = CUDA_SUCCESS;
		if (contactCount)
			contactsResult = cudaContext->memcpyDtoH(contacts.data(), sortedContactsd,
				size_t(contactCount) * sizeof(PxgParticlePrimitiveContact));
		if (pairCount)
		{
			startsResult = cudaContext->memcpyDtoH(rangeStarts.data(), rangeStartd, size_t(pairCount) * sizeof(PxU32));
			endsResult = cudaContext->memcpyDtoH(rangeEnds.data(), rangeEndd, size_t(pairCount) * sizeof(PxU32));
		}

		if (contactsResult != CUDA_SUCCESS || startsResult != CUDA_SUCCESS || endsResult != CUDA_SUCCESS)
		{
			std::printf("[DCU PBD RANGE DIAG] marker=PX_DCU_PBD_PARTICLE_RANGE_DIAG_V60 verdict=READBACK_FAILED contacts_copy=%d starts_copy=%d ends_copy=%d contacts=%u pairs=%u\n",
				int(contactsResult), int(startsResult), int(endsResult), contactCount, pairCount);
			std::fflush(stdout);
			return false;
		}

		std::vector<PxU32> expectedStarts;
		std::vector<PxU32> expectedEnds;
		expectedStarts.reserve(contactCount);
		expectedEnds.reserve(contactCount);
		PxU32 orderViolations = 0;
		for (PxU32 i = 0; i < contactCount; ++i)
		{
			if (i && contacts[i - 1].particleId > contacts[i].particleId)
				++orderViolations;
			if (i == 0 || contacts[i - 1].particleId != contacts[i].particleId)
			{
				if (i)
					expectedEnds.push_back(i);
				expectedStarts.push_back(i);
			}
		}
		if (contactCount)
			expectedEnds.push_back(contactCount);

		const PxU32 expectedPairs = PxU32(expectedStarts.size());
		const PxU32 pairCountMismatch = pairCount == expectedPairs ? 0u : 1u;
		PxU32 startMismatches = 0;
		PxU32 endMismatches = 0;
		PxU32 firstMismatch = 0xffffffffu;
		if (pairCount == expectedPairs)
		{
			for (PxU32 i = 0; i < pairCount; ++i)
			{
				const bool startMismatch = rangeStarts[i] != expectedStarts[i];
				const bool endMismatch = rangeEnds[i] != expectedEnds[i];
				startMismatches += startMismatch ? 1u : 0u;
				endMismatches += endMismatch ? 1u : 0u;
				if ((startMismatch || endMismatch) && firstMismatch == 0xffffffffu)
					firstMismatch = i;
			}
		}

		const bool valid = pairCountMismatch == 0 && startMismatches == 0 && endMismatches == 0 && orderViolations == 0;
		const PxU32 expectedStart = firstMismatch == 0xffffffffu ? 0xffffffffu : expectedStarts[firstMismatch];
		const PxU32 expectedEnd = firstMismatch == 0xffffffffu ? 0xffffffffu : expectedEnds[firstMismatch];
		const PxU32 actualStart = firstMismatch == 0xffffffffu ? 0xffffffffu : rangeStarts[firstMismatch];
		const PxU32 actualEnd = firstMismatch == 0xffffffffu ? 0xffffffffu : rangeEnds[firstMismatch];
		std::printf("[DCU PBD RANGE DIAG] marker=PX_DCU_PBD_PARTICLE_RANGE_DIAG_V60 contacts=%u pairs=%u expected_pairs=%u pair_count_mismatch=%u start_mismatches=%u end_mismatches=%u particle_order_violations=%u first_mismatch=%u expected_start=%u expected_end=%u actual_start=%u actual_end=%u verdict=%s\n",
			contactCount, pairCount, expectedPairs, pairCountMismatch, startMismatches, endMismatches, orderViolations, firstMismatch,
			expectedStart, expectedEnd, actualStart, actualEnd, valid ? "VALID" : "INVALID");
		std::fflush(stdout);
		return valid;
	}

	void dcuPbdPrintPrimitiveRadixModeOnce(bool chunked, bool inlineRadix, bool fused64)
	{
		static std::atomic<bool> printed(false);
		if(!printed.exchange(true, std::memory_order_relaxed))
		{
			std::printf("[DCU PBD RADIX] host_marker=%s mode=%s device_marker=%s launches_per_64bit_sort=%u launch_grid=1x2x1 launch_block=%ux1x1\n",
				fused64 ? "PX_DCU_PBD_PRIMITIVE_RADIX_HOST_V63_FUSED64_OPT_IN" : (inlineRadix ? "PX_DCU_PBD_PRIMITIVE_RADIX_HOST_V7_INLINE_OPT_IN" : "PX_DCU_PBD_PRIMITIVE_RADIX_HOST_V5_OPT_IN"),
				fused64 ? "fused64_v63" : (inlineRadix ? "inline_v7" : (chunked ? "chunked_v5" : "serial_v4")),
				fused64 ? "PX_DCU_PBD_PRIMITIVE_RADIX_V63_FUSED_64_INLINE" : (inlineRadix ? "PX_DCU_PBD_PRIMITIVE_RADIX_V7_INLINE_RADIX_32" : (chunked ? "PX_DCU_PBD_PRIMITIVE_RADIX_V5_CHUNKED_COUNTING_32" : "PX_DCU_PBD_PRIMITIVE_RADIX_V4_SERIAL_COUNTING_4096")),
				fused64 ? 1u : (chunked ? 2u : 16u), chunked ? 32u : 1u);
			std::fflush(stdout);
		}
	}

	struct DcuPbdPrimitiveSortItem
	{
		PxU32 key;
		PxU32 rank;
	};

	struct DcuPbdPrimitiveSortItem64
	{
		PxU64 key;
		PxU32 rank;
	};

	struct DcuPbdPrimitiveSortReference
	{
		bool enabled;
		PxU32 count;
		std::vector<DcuPbdPrimitiveSortItem> rigid;
		std::vector<DcuPbdPrimitiveSortItem> particle;
		std::vector<DcuPbdPrimitiveSortItem64> rigid64;
		std::vector<DcuPbdPrimitiveSortItem64> particle64;

		DcuPbdPrimitiveSortReference() : enabled(false), count(0)
		{
		}
	};

	struct DcuPbdRankStats
	{
		PxU32 minRank;
		PxU32 maxRank;
		PxU32 outOfRangeCount;
		PxU32 duplicateCount;
		PxU32 missingCount;
		PxU32 firstInvalidWorkIndex;
		PxU32 firstInvalidValue;
		PxU32 firstDuplicateWorkIndex;
		PxU32 firstDuplicateValue;
	};

	DcuPbdRankStats dcuPbdAnalyzeRanks(const std::vector<PxU32>& ranks)
	{
		DcuPbdRankStats stats;
		stats.minRank = 0xffffffffu;
		stats.maxRank = 0;
		stats.outOfRangeCount = 0;
		stats.duplicateCount = 0;
		stats.missingCount = 0;
		stats.firstInvalidWorkIndex = 0xffffffffu;
		stats.firstInvalidValue = 0xffffffffu;
		stats.firstDuplicateWorkIndex = 0xffffffffu;
		stats.firstDuplicateValue = 0xffffffffu;

		std::vector<PxU8> seen(ranks.size(), PxU8(0));
		for (PxU32 i = 0; i < ranks.size(); ++i)
		{
			const PxU32 rank = ranks[i];
			stats.minRank = PxMin(stats.minRank, rank);
			stats.maxRank = PxMax(stats.maxRank, rank);

			if (rank >= ranks.size())
			{
				++stats.outOfRangeCount;
				if (stats.firstInvalidWorkIndex == 0xffffffffu)
				{
					stats.firstInvalidWorkIndex = i;
					stats.firstInvalidValue = rank;
				}
			}
			else if (seen[rank])
			{
				++stats.duplicateCount;
				if (stats.firstDuplicateWorkIndex == 0xffffffffu)
				{
					stats.firstDuplicateWorkIndex = i;
					stats.firstDuplicateValue = rank;
				}
			}
			else
				seen[rank] = PxU8(1);
		}

		for (PxU32 i = 0; i < seen.size(); ++i)
			if (!seen[i])
				++stats.missingCount;

		return stats;
	}

	bool dcuPbdCapturePrimitiveSortReference(PxCudaContext* cudaContext, CUstream stream, CUdeviceptr deviceCount,
		CUdeviceptr rigidKeys, CUdeviceptr rigidRanks, CUdeviceptr particleKeys, CUdeviceptr particleRanks,
		CUdeviceptr rigidKeys64, CUdeviceptr particleKeys64,
		PxU32 maxCount, DcuPbdPrimitiveSortReference& reference)
	{
		if (!dcuPbdPrimitiveSortDiagEnabled())
			return true;

		reference.enabled = true;
		const PxCUresult syncResult = cudaContext->streamSynchronize(stream);
		reference.count = 0;
		const PxCUresult countCopyResult = syncResult == CUDA_SUCCESS
			? cudaContext->memcpyDtoH(&reference.count, deviceCount, sizeof(reference.count)) : syncResult;
		if (syncResult != CUDA_SUCCESS || countCopyResult != CUDA_SUCCESS)
		{
			std::printf("[DCU PBD SORT] marker=PX_DCU_PBD_PRIMITIVE_SORT_DIAG_V4 phase=pre_low32 verdict=READBACK_FAILED sync=%d count_copy=%d count=%u action=abort_before_low32_sort\n",
				int(syncResult), int(countCopyResult), reference.count);
			std::fflush(stdout);
			return false;
		}
		if (reference.count > maxCount)
		{
			std::printf("[DCU PBD SORT] marker=PX_DCU_PBD_PRIMITIVE_SORT_DIAG_V4 phase=pre_low32 verdict=COUNT_OUT_OF_RANGE count=%u max_count=%u action=abort_before_low32_sort\n",
				reference.count, maxCount);
			std::fflush(stdout);
			return false;
		}

		std::vector<PxU32> rigidInputKeys(reference.count);
		std::vector<PxU32> rigidInputRanks(reference.count);
		std::vector<PxU32> particleInputKeys(reference.count);
		std::vector<PxU32> particleInputRanks(reference.count);
		std::vector<PxU64> rigidInputKeys64(reference.count);
		std::vector<PxU64> particleInputKeys64(reference.count);
		PxCUresult rigidKeyCopyResult = CUDA_SUCCESS;
		PxCUresult rigidRankCopyResult = CUDA_SUCCESS;
		PxCUresult particleKeyCopyResult = CUDA_SUCCESS;
		PxCUresult particleRankCopyResult = CUDA_SUCCESS;
		PxCUresult rigidKey64CopyResult = CUDA_SUCCESS;
		PxCUresult particleKey64CopyResult = CUDA_SUCCESS;
		if (reference.count)
		{
			const size_t bytes = sizeof(PxU32) * reference.count;
			rigidKeyCopyResult = cudaContext->memcpyDtoH(rigidInputKeys.data(), rigidKeys, bytes);
			rigidRankCopyResult = cudaContext->memcpyDtoH(rigidInputRanks.data(), rigidRanks, bytes);
			particleKeyCopyResult = cudaContext->memcpyDtoH(particleInputKeys.data(), particleKeys, bytes);
			particleRankCopyResult = cudaContext->memcpyDtoH(particleInputRanks.data(), particleRanks, bytes);
			rigidKey64CopyResult = cudaContext->memcpyDtoH(rigidInputKeys64.data(), rigidKeys64,
				size_t(reference.count) * sizeof(PxU64));
			particleKey64CopyResult = cudaContext->memcpyDtoH(particleInputKeys64.data(), particleKeys64,
				size_t(reference.count) * sizeof(PxU64));
		}
		if (rigidKeyCopyResult != CUDA_SUCCESS || rigidRankCopyResult != CUDA_SUCCESS ||
			particleKeyCopyResult != CUDA_SUCCESS || particleRankCopyResult != CUDA_SUCCESS ||
			rigidKey64CopyResult != CUDA_SUCCESS || particleKey64CopyResult != CUDA_SUCCESS)
		{
			std::printf("[DCU PBD SORT] marker=PX_DCU_PBD_PRIMITIVE_SORT_DIAG_V4 phase=pre_low32 verdict=READBACK_FAILED sync=%d count_copy=%d rigid_key_copy=%d rigid_rank_copy=%d particle_key_copy=%d particle_rank_copy=%d rigid_key64_copy=%d particle_key64_copy=%d count=%u action=abort_before_sort\n",
				int(syncResult), int(countCopyResult), int(rigidKeyCopyResult), int(rigidRankCopyResult),
				int(particleKeyCopyResult), int(particleRankCopyResult), int(rigidKey64CopyResult),
				int(particleKey64CopyResult), reference.count);
			std::fflush(stdout);
			return false;
		}

		reference.rigid.resize(reference.count);
		reference.particle.resize(reference.count);
		reference.rigid64.resize(reference.count);
		reference.particle64.resize(reference.count);
		for (PxU32 i = 0; i < reference.count; ++i)
		{
			reference.rigid[i].key = rigidInputKeys[i];
			reference.rigid[i].rank = rigidInputRanks[i];
			reference.particle[i].key = particleInputKeys[i];
			reference.particle[i].rank = particleInputRanks[i];
			if (rigidInputRanks[i] >= reference.count || particleInputRanks[i] >= reference.count)
			{
				std::printf("[DCU PBD SORT] marker=PX_DCU_PBD_PRIMITIVE_SORT_DIAG_V63 phase=pre_64 verdict=INPUT_RANK_OUT_OF_RANGE count=%u index=%u rigid_rank=%u particle_rank=%u action=abort_before_sort\n",
					reference.count, i, rigidInputRanks[i], particleInputRanks[i]);
				std::fflush(stdout);
				return false;
			}
			reference.rigid64[i].key = rigidInputKeys64[rigidInputRanks[i]];
			reference.rigid64[i].rank = rigidInputRanks[i];
			reference.particle64[i].key = particleInputKeys64[particleInputRanks[i]];
			reference.particle64[i].rank = particleInputRanks[i];
		}
		const auto keyLess = [](const DcuPbdPrimitiveSortItem& a, const DcuPbdPrimitiveSortItem& b)
		{
			return a.key < b.key;
		};
		std::stable_sort(reference.rigid.begin(), reference.rigid.end(), keyLess);
		std::stable_sort(reference.particle.begin(), reference.particle.end(), keyLess);
		const auto keyLess64 = [](const DcuPbdPrimitiveSortItem64& a, const DcuPbdPrimitiveSortItem64& b)
		{
			return a.key < b.key;
		};
		std::stable_sort(reference.rigid64.begin(), reference.rigid64.end(), keyLess64);
		std::stable_sort(reference.particle64.begin(), reference.particle64.end(), keyLess64);

		std::printf("[DCU PBD SORT] marker=PX_DCU_PBD_PRIMITIVE_SORT_DIAG_V4 phase=pre_low32 verdict=REFERENCE_READY count=%u reference=host_stable_sort_key_rank\n",
			reference.count);
		std::fflush(stdout);
		return true;
	}

	bool dcuPbdValidatePrimitiveSortOutput(PxCudaContext* cudaContext, CUstream stream, CUdeviceptr deviceCount,
		CUdeviceptr rigidKeys, CUdeviceptr rigidRanks, CUdeviceptr particleKeys, CUdeviceptr particleRanks,
		const DcuPbdPrimitiveSortReference& reference)
	{
		if (!reference.enabled)
			return true;

		const PxCUresult syncResult = cudaContext->streamSynchronize(stream);
		PxU32 outputCount = 0;
		const PxCUresult countCopyResult = syncResult == CUDA_SUCCESS
			? cudaContext->memcpyDtoH(&outputCount, deviceCount, sizeof(outputCount)) : syncResult;
		if (syncResult != CUDA_SUCCESS || countCopyResult != CUDA_SUCCESS || outputCount != reference.count)
		{
			std::printf("[DCU PBD SORT] marker=PX_DCU_PBD_PRIMITIVE_SORT_DIAG_V4 phase=post_low32 verdict=%s sync=%d count_copy=%d input_count=%u output_count=%u action=abort_before_high32_copy\n",
				(syncResult == CUDA_SUCCESS && countCopyResult == CUDA_SUCCESS) ? "COUNT_CHANGED" : "READBACK_FAILED",
				int(syncResult), int(countCopyResult), reference.count, outputCount);
			std::fflush(stdout);
			return false;
		}

		std::vector<PxU32> rigidOutputKeys(outputCount);
		std::vector<PxU32> rigidOutputRanks(outputCount);
		std::vector<PxU32> particleOutputKeys(outputCount);
		std::vector<PxU32> particleOutputRanks(outputCount);
		PxCUresult rigidKeyCopyResult = CUDA_SUCCESS;
		PxCUresult rigidRankCopyResult = CUDA_SUCCESS;
		PxCUresult particleKeyCopyResult = CUDA_SUCCESS;
		PxCUresult particleRankCopyResult = CUDA_SUCCESS;
		if (outputCount)
		{
			const size_t bytes = sizeof(PxU32) * outputCount;
			rigidKeyCopyResult = cudaContext->memcpyDtoH(rigidOutputKeys.data(), rigidKeys, bytes);
			rigidRankCopyResult = cudaContext->memcpyDtoH(rigidOutputRanks.data(), rigidRanks, bytes);
			particleKeyCopyResult = cudaContext->memcpyDtoH(particleOutputKeys.data(), particleKeys, bytes);
			particleRankCopyResult = cudaContext->memcpyDtoH(particleOutputRanks.data(), particleRanks, bytes);
		}
		if (rigidKeyCopyResult != CUDA_SUCCESS || rigidRankCopyResult != CUDA_SUCCESS ||
			particleKeyCopyResult != CUDA_SUCCESS || particleRankCopyResult != CUDA_SUCCESS)
		{
			std::printf("[DCU PBD SORT] marker=PX_DCU_PBD_PRIMITIVE_SORT_DIAG_V4 phase=post_low32 verdict=READBACK_FAILED rigid_key_copy=%d rigid_rank_copy=%d particle_key_copy=%d particle_rank_copy=%d count=%u action=abort_before_high32_copy\n",
				int(rigidKeyCopyResult), int(rigidRankCopyResult), int(particleKeyCopyResult),
				int(particleRankCopyResult), outputCount);
			std::fflush(stdout);
			return false;
		}

		bool allValid = true;
		for (PxU32 setIndex = 0; setIndex < 2; ++setIndex)
		{
			const char* label = setIndex == 0 ? "rigid" : "particle";
			const std::vector<DcuPbdPrimitiveSortItem>& expected = setIndex == 0 ? reference.rigid : reference.particle;
			const std::vector<PxU32>& actualKeys = setIndex == 0 ? rigidOutputKeys : particleOutputKeys;
			const std::vector<PxU32>& actualRanks = setIndex == 0 ? rigidOutputRanks : particleOutputRanks;
			PxU32 keyMismatches = 0;
			PxU32 rankMismatches = 0;
			PxU32 pairMismatches = 0;
			PxU32 orderViolations = 0;
			PxU32 firstMismatch = 0xffffffffu;
			for (PxU32 i = 0; i < outputCount; ++i)
			{
				const bool keyMismatch = actualKeys[i] != expected[i].key;
				const bool rankMismatch = actualRanks[i] != expected[i].rank;
				keyMismatches += keyMismatch ? 1u : 0u;
				rankMismatches += rankMismatch ? 1u : 0u;
				if (keyMismatch || rankMismatch)
				{
					++pairMismatches;
					if (firstMismatch == 0xffffffffu)
						firstMismatch = i;
				}
				if (i && actualKeys[i - 1] > actualKeys[i])
					++orderViolations;
			}

			const DcuPbdRankStats rankStats = dcuPbdAnalyzeRanks(actualRanks);
			const bool valid = pairMismatches == 0 && orderViolations == 0 && rankStats.outOfRangeCount == 0 &&
				rankStats.duplicateCount == 0 && rankStats.missingCount == 0;
			const PxU32 expectedKey = firstMismatch == 0xffffffffu ? 0xffffffffu : expected[firstMismatch].key;
			const PxU32 expectedRank = firstMismatch == 0xffffffffu ? 0xffffffffu : expected[firstMismatch].rank;
			const PxU32 actualKey = firstMismatch == 0xffffffffu ? 0xffffffffu : actualKeys[firstMismatch];
			const PxU32 actualRank = firstMismatch == 0xffffffffu ? 0xffffffffu : actualRanks[firstMismatch];
			std::printf("[DCU PBD SORT] marker=PX_DCU_PBD_PRIMITIVE_SORT_DIAG_V4 phase=post_low32 label=%s count=%u key_mismatches=%u rank_mismatches=%u pair_mismatches=%u key_order_violations=%u rank_out_of_range=%u rank_duplicates=%u rank_missing=%u first_mismatch=%u expected_key=%u expected_rank=%u actual_key=%u actual_rank=%u verdict=%s\n",
				label, outputCount, keyMismatches, rankMismatches, pairMismatches, orderViolations,
				rankStats.outOfRangeCount, rankStats.duplicateCount, rankStats.missingCount, firstMismatch,
				expectedKey, expectedRank, actualKey, actualRank, valid ? "VALID" : "INVALID");
			allValid = allValid && valid;
		}

		const bool chunked = dcuPbdUseChunkedPrimitiveRadix();
		const bool inlineRadix = chunked && dcuPbdUseInlinePrimitiveRadix();
		std::printf("[DCU PBD SORT] marker=PX_DCU_PBD_PRIMITIVE_SORT_DIAG_V4 phase=post_low32 verdict=%s count=%u reference=host_stable_sort_key_rank action=%s radix_mode=%s device_marker=%s launch_grid=1x2x1 launch_block=%ux1x1\n",
			allValid ? "VALID" : "INVALID", outputCount,
			allValid ? "continue_high32_copy" : "abort_before_high32_copy",
			inlineRadix ? "inline_v7" : (chunked ? "chunked_v5" : "serial_v4"),
			inlineRadix ? "PX_DCU_PBD_PRIMITIVE_RADIX_V7_INLINE_RADIX_32" : (chunked ? "PX_DCU_PBD_PRIMITIVE_RADIX_V5_CHUNKED_COUNTING_32" : "PX_DCU_PBD_PRIMITIVE_RADIX_V4_SERIAL_COUNTING_4096"),
			chunked ? 32u : 1u);
		std::fflush(stdout);
		return allValid;
	}

	bool dcuPbdValidatePrimitiveSortOutput64(PxCudaContext* cudaContext, CUstream stream, CUdeviceptr deviceCount,
		CUdeviceptr rigidRanks, CUdeviceptr particleRanks, CUdeviceptr rigidKeys64, CUdeviceptr particleKeys64,
		const DcuPbdPrimitiveSortReference& reference)
	{
		if (!reference.enabled)
			return true;

		const PxCUresult syncResult = cudaContext->streamSynchronize(stream);
		PxU32 outputCount = 0;
		const PxCUresult countCopyResult = syncResult == CUDA_SUCCESS
			? cudaContext->memcpyDtoH(&outputCount, deviceCount, sizeof(outputCount)) : syncResult;
		if (syncResult != CUDA_SUCCESS || countCopyResult != CUDA_SUCCESS || outputCount != reference.count)
		{
			std::printf("[DCU PBD SORT] marker=PX_DCU_PBD_PRIMITIVE_SORT_DIAG_V63 phase=post_64 verdict=%s sync=%d count_copy=%d input_count=%u output_count=%u action=abort_before_sorted_value_copy\n",
				(syncResult == CUDA_SUCCESS && countCopyResult == CUDA_SUCCESS) ? "COUNT_CHANGED" : "READBACK_FAILED",
				int(syncResult), int(countCopyResult), reference.count, outputCount);
			std::fflush(stdout);
			return false;
		}

		std::vector<PxU32> rigidOutputRanks(outputCount);
		std::vector<PxU32> particleOutputRanks(outputCount);
		std::vector<PxU64> rigidInputKeys64(outputCount);
		std::vector<PxU64> particleInputKeys64(outputCount);
		PxCUresult rigidRankCopyResult = CUDA_SUCCESS;
		PxCUresult particleRankCopyResult = CUDA_SUCCESS;
		PxCUresult rigidKey64CopyResult = CUDA_SUCCESS;
		PxCUresult particleKey64CopyResult = CUDA_SUCCESS;
		if (outputCount)
		{
			rigidRankCopyResult = cudaContext->memcpyDtoH(rigidOutputRanks.data(), rigidRanks,
				size_t(outputCount) * sizeof(PxU32));
			particleRankCopyResult = cudaContext->memcpyDtoH(particleOutputRanks.data(), particleRanks,
				size_t(outputCount) * sizeof(PxU32));
			rigidKey64CopyResult = cudaContext->memcpyDtoH(rigidInputKeys64.data(), rigidKeys64,
				size_t(outputCount) * sizeof(PxU64));
			particleKey64CopyResult = cudaContext->memcpyDtoH(particleInputKeys64.data(), particleKeys64,
				size_t(outputCount) * sizeof(PxU64));
		}
		if (rigidRankCopyResult != CUDA_SUCCESS || particleRankCopyResult != CUDA_SUCCESS ||
			rigidKey64CopyResult != CUDA_SUCCESS || particleKey64CopyResult != CUDA_SUCCESS)
		{
			std::printf("[DCU PBD SORT] marker=PX_DCU_PBD_PRIMITIVE_SORT_DIAG_V63 phase=post_64 verdict=READBACK_FAILED rigid_rank_copy=%d particle_rank_copy=%d rigid_key64_copy=%d particle_key64_copy=%d count=%u action=abort_before_sorted_value_copy\n",
				int(rigidRankCopyResult), int(particleRankCopyResult), int(rigidKey64CopyResult),
				int(particleKey64CopyResult), outputCount);
			std::fflush(stdout);
			return false;
		}

		bool allValid = true;
		for (PxU32 setIndex = 0; setIndex < 2; ++setIndex)
		{
			const char* label = setIndex == 0 ? "rigid" : "particle";
			const std::vector<DcuPbdPrimitiveSortItem64>& expected = setIndex == 0 ? reference.rigid64 : reference.particle64;
			const std::vector<PxU32>& actualRanks = setIndex == 0 ? rigidOutputRanks : particleOutputRanks;
			const std::vector<PxU64>& originalKeys = setIndex == 0 ? rigidInputKeys64 : particleInputKeys64;
			const DcuPbdRankStats rankStats = dcuPbdAnalyzeRanks(actualRanks);
			PxU32 keyMismatches = 0;
			PxU32 rankMismatches = 0;
			PxU32 pairMismatches = 0;
			PxU32 orderViolations = 0;
			PxU32 firstMismatch = 0xffffffffu;
			PxU64 previousKey = 0;
			bool previousKeyValid = false;
			for (PxU32 i = 0; i < outputCount; ++i)
			{
				const bool rankInRange = actualRanks[i] < outputCount;
				const PxU64 actualKey = rankInRange ? originalKeys[actualRanks[i]] : 0xffffffffffffffffull;
				const bool keyMismatch = !rankInRange || actualKey != expected[i].key;
				const bool rankMismatch = actualRanks[i] != expected[i].rank;
				keyMismatches += keyMismatch ? 1u : 0u;
				rankMismatches += rankMismatch ? 1u : 0u;
				if (keyMismatch || rankMismatch)
				{
					++pairMismatches;
					if (firstMismatch == 0xffffffffu)
						firstMismatch = i;
				}
				if (rankInRange && previousKeyValid && previousKey > actualKey)
					++orderViolations;
				previousKey = actualKey;
				previousKeyValid = rankInRange;
			}

			const bool valid = pairMismatches == 0 && orderViolations == 0 && rankStats.outOfRangeCount == 0 &&
				rankStats.duplicateCount == 0 && rankStats.missingCount == 0;
			const PxU64 expectedKey = firstMismatch == 0xffffffffu ? 0xffffffffffffffffull : expected[firstMismatch].key;
			const PxU32 expectedRank = firstMismatch == 0xffffffffu ? 0xffffffffu : expected[firstMismatch].rank;
			const PxU32 actualRank = firstMismatch == 0xffffffffu ? 0xffffffffu : actualRanks[firstMismatch];
			const PxU64 actualKey = firstMismatch == 0xffffffffu || actualRank >= outputCount
				? 0xffffffffffffffffull : originalKeys[actualRank];
			std::printf("[DCU PBD SORT] marker=PX_DCU_PBD_PRIMITIVE_SORT_DIAG_V63 phase=post_64 label=%s count=%u key64_mismatches=%u rank_mismatches=%u pair_mismatches=%u key64_order_violations=%u rank_out_of_range=%u rank_duplicates=%u rank_missing=%u first_mismatch=%u expected_key64=%llu expected_rank=%u actual_key64=%llu actual_rank=%u verdict=%s\n",
				label, outputCount, keyMismatches, rankMismatches, pairMismatches, orderViolations,
				rankStats.outOfRangeCount, rankStats.duplicateCount, rankStats.missingCount, firstMismatch,
				static_cast<unsigned long long>(expectedKey), expectedRank,
				static_cast<unsigned long long>(actualKey), actualRank, valid ? "VALID" : "INVALID");
			allValid = allValid && valid;
		}

		const bool fused64 = dcuPbdUseChunkedPrimitiveRadix() && dcuPbdUseInlinePrimitiveRadix() &&
			dcuPbdUseFused64PrimitiveRadix();
		std::printf("[DCU PBD SORT] marker=PX_DCU_PBD_PRIMITIVE_SORT_DIAG_V63 phase=post_64 verdict=%s count=%u reference=host_stable_sort_key64_rank action=%s radix_mode=%s device_marker=%s launch_grid=1x2x1 launch_block=32x1x1\n",
			allValid ? "VALID" : "INVALID", outputCount,
			allValid ? "continue_sorted_value_copy" : "abort_before_sorted_value_copy",
			fused64 ? "fused64_v63" : "legacy_64bit_two_stage",
			fused64 ? "PX_DCU_PBD_PRIMITIVE_RADIX_V63_FUSED_64_INLINE" : "legacy_low32_high32_sequence");
		std::fflush(stdout);
		return allValid;
	}
	#endif

	namespace
	{
		// Generates an optimally dense sphere packing at the origin (implicit sphere at the origin)
		inline PxU32 TightPack3D(PxReal radius, PxReal separation, PxVec3* points, PxU32 maxPoints)
		{
			if (separation <= 0.0f)
			{
				return 0;
			}

			PxI32 dim = PxI32(ceilf(radius / separation));
			PxU32 c = 0;

			for (PxI32 z = -dim; z <= dim; ++z)
			{
				for (PxI32 y = -dim; y <= dim; ++y)
				{
					for (PxI32 x = -dim; x <= dim; ++x)
					{
						PxReal xpos = x * separation + (((y + z) & 1) ? separation * 0.5f : 0.0f);
						PxReal ypos = y * sqrtf(0.75f)*separation;
						PxReal zpos = z * sqrtf(0.75f)*separation;

						PxVec3 p(xpos, ypos, zpos);

						// skip center
						if (p.magnitudeSquared() == 0.0f)
							continue;

						if (c == maxPoints)
							return c;

						if (p.magnitude() <= radius)
						{
							points[c] = p;
							++c;
						}
					}
				}
			}

			return c;
		}

		const PxReal kPi = 3.141592653589f;

		inline float sqr(float x) { return x * x; }

		inline float W(float x, float h)
		{
			// Clavet et al. unnormalized spike
			const float k = 15.0f / (kPi*h*h*h);
			return k * sqr(1.0f - x / h);
		}

		inline float dWdx(float x, float h)
		{
			const float k = 15.0f / (kPi*h*h*h);
			return -k * 2.0f*(1.0f - x / h) / h;
		}

		// calculate the rest density for given radius and rest distance
		// rho is the rest density, rhoDeriv is the scaling factor for the density constraint
		inline void CalculateRestDensity(float restDistance, float h, float& rho, float& rhoDeriv, float& surfaceDeriv)
		{
			PxVec3 samples[2048];

			PxU32 n = TightPack3D(h, restDistance, &samples[0], 2048);

			rho = 0.0f;
			rhoDeriv = 0.0f;

			float a = 0.0f;
			float b = 0.0f;

			// calculate rest density given fixed samples
			for (PxU32 i = 0; i < n; ++i)
			{
				float d = samples[i].magnitude();

				rho += W(d, h);
				rhoDeriv += sqr(dWdx(d, h));

				if (samples[i].y <= 0.0f)
				{
					float cosTheta = samples[i].y / samples[i].magnitude();

					a += dWdx(d, h)*cosTheta;
					b -= samples[i].magnitude() * cosTheta;
				}
			}

			surfaceDeriv = a / b;
		}
	}
	extern "C" void initParticleSystemKernels0();
	extern "C" void initDiffuseParticlesKernels0();
	extern "C" void initIsosurfaceExtractionKernels0();
	extern "C" void initSparseGridStandaloneKernels0();
	extern "C" void initAnisotropyKernels0();
	extern "C" void initAlgorithmsKernels0();
	extern "C" void initSdfConstructionKernels0();

	void createPxgParticleSystem()
	{
#if !PX_PHYSX_GPU_EXPORTS
		//this call is needed to force PhysXSimulationControllerGpu linkage as Static Library!
		initParticleSystemKernels0();
		initDiffuseParticlesKernels0();
		initIsosurfaceExtractionKernels0();
		initSparseGridStandaloneKernels0();
		initAnisotropyKernels0();
		initAlgorithmsKernels0();
		initSdfConstructionKernels0();
#endif
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////

	static PxU32 uniqueIdIndexer = 0;

	template<class BufferClass>
	PxgParticleBufferBase<BufferClass>::PxgParticleBufferBase(PxU32 maxNumParticles, PxU32 maxNumVolumes, PxCudaContextManager& contextManager) :
		mBufferFlags(0),
		mContextManager(contextManager),
		mPositionInvMassesD(NULL),
		mVelocitiesD(NULL),
		mPhasesD(NULL),
		mVolumesD(NULL),
		mPositionInvMassesH(NULL),
		mVelocitiesH(NULL),
		mPhasesH(NULL),
		mVolumesH(NULL),
		mFilterPairs(NULL),
		mRigidAttachments(NULL),
		mNumActiveParticles(0),
		mMaxNumParticles(maxNumParticles),
		mNumParticleVolumes(0),
		mMaxNumVolumes(maxNumVolumes),
		mNumFilterPairs(0),
		mNumRigidAttachments(0),
		mFlatListStartIndex(0),
		mUniqueId(uniqueIdIndexer++)
	{
		PX_ASSERT(maxNumParticles > 0);
		mPositionInvMassesD = PX_DEVICE_MEMORY_ALLOC(PxVec4, contextManager, maxNumParticles);
		mVelocitiesD = PX_DEVICE_MEMORY_ALLOC(PxVec4, contextManager, maxNumParticles);
		mPhasesD = PX_DEVICE_MEMORY_ALLOC(PxU32, contextManager, maxNumParticles);
		if (maxNumVolumes)
		{
			mVolumesD = PX_DEVICE_MEMORY_ALLOC(PxParticleVolume, contextManager, maxNumVolumes);
		}
	}

	template<class BufferClass>
	PxgParticleBufferBase<BufferClass>::~PxgParticleBufferBase()
	{
		PX_DEVICE_MEMORY_FREE(mContextManager, mPositionInvMassesD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mVelocitiesD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mPhasesD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mVolumesD);
		
		PX_PINNED_MEMORY_FREE(mContextManager, mPositionInvMassesH);
		PX_PINNED_MEMORY_FREE(mContextManager, mVelocitiesH);
		PX_PINNED_MEMORY_FREE(mContextManager, mPhasesH);
		PX_PINNED_MEMORY_FREE(mContextManager, mVolumesH);
	}

	template<class BufferClass>
	void PxgParticleBufferBase<BufferClass>::setNbActiveParticles(PxU32 numActiveParticles)
	{
		PX_ASSERT(numActiveParticles <= mMaxNumParticles);
		if (mNumActiveParticles != numActiveParticles)
		{
			mBufferFlags |= (PxParticleBufferFlag::eUPDATE_POSITION | PxParticleBufferFlag::eUPDATE_VELOCITY |
				PxParticleBufferFlag::eUPDATE_PHASE);
		}
		mNumActiveParticles = numActiveParticles;
	}

	template<class BufferClass>
	void PxgParticleBufferBase<BufferClass>::setRigidFilters(PxParticleRigidFilterPair* filters, PxU32 numFilters)
	{
		mFilterPairs = filters;
		mNumFilterPairs = numFilters;

		mBufferFlags |= PxParticleBufferFlag::eUPDATE_ATTACHMENTS;
	}

	template<class BufferClass>
	void PxgParticleBufferBase<BufferClass>::setRigidAttachments(PxParticleRigidAttachment* attachments, PxU32 numAttachments)
	{
		mRigidAttachments = attachments;
		mNumRigidAttachments = numAttachments;

		mBufferFlags |= PxParticleBufferFlag::eUPDATE_ATTACHMENTS;
	}

	template<class BufferClass>
	void PxgParticleBufferBase<BufferClass>::allocHostBuffers()
	{
		PX_ASSERT(mMaxNumParticles > 0);
		if (!mPositionInvMassesH)
		{
			PX_ASSERT(!mVelocitiesH && !mPhasesH && !mVolumesH);
			mPositionInvMassesH = PX_PINNED_MEMORY_ALLOC(PxVec4, mContextManager, mMaxNumParticles);
			mVelocitiesH = PX_PINNED_MEMORY_ALLOC(PxVec4, mContextManager, mMaxNumParticles);
			mPhasesH = PX_PINNED_MEMORY_ALLOC(PxU32, mContextManager, mMaxNumParticles);
			if (mMaxNumVolumes)
			{
				mVolumesH = PX_PINNED_MEMORY_ALLOC(PxParticleVolume, mContextManager, mMaxNumVolumes);
			}
		}
	}

	template<class BufferClass>
	void PxgParticleBufferBase<BufferClass>::copyToHost(CUstream stream)
	{
		mContextManager.copyDToHAsync<PxVec4>(mPositionInvMassesH, mPositionInvMassesD, mNumActiveParticles, stream);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	PxgParticleBuffer::PxgParticleBuffer(PxU32 maxNumParticles, PxU32 maxNumVolumes, PxCudaContextManager& contextManager) :
		PxgParticleBufferBase<PxsParticleBuffer>(maxNumParticles, maxNumVolumes, contextManager)
	{}

	void PxgParticleBuffer::copyToHost(CUstream stream)
	{
		PxgParticleBufferBase<PxsParticleBuffer>::copyToHost(stream);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	PxgParticleAndDiffuseBuffer::PxgParticleAndDiffuseBuffer(PxU32 maxNumParticles, PxU32 maxNumVolumes, PxU32 maxNumDiffuseParticles, PxCudaContextManager& contextManager) :
		PxgParticleBufferBase<PxsParticleAndDiffuseBuffer>(maxNumParticles, maxNumVolumes, contextManager),
		mDiffusePositionsLifeTimeD(NULL),
		mDiffuseVelocitiesD(NULL),
		mNumDiffuseParticlesD(NULL),
		mMaxNumDiffuseParticles(0),
		mMaxActiveDiffuseParticles(0),
		mNumActiveDiffuseParticlesH(NULL)
	{
		mContextManager.acquireContext();

		if (maxNumDiffuseParticles > 0)
		{
			mDiffusePositionsLifeTimeD = PX_DEVICE_MEMORY_ALLOC(PxVec4, mContextManager, maxNumDiffuseParticles);
			mDiffuseVelocitiesD = PX_DEVICE_MEMORY_ALLOC(PxVec4, mContextManager, maxNumDiffuseParticles);
			mMaxActiveDiffuseParticles = maxNumDiffuseParticles;
		}

		mNumDiffuseParticlesD = PX_DEVICE_MEMORY_ALLOC(int, mContextManager, 2);
		mNumActiveDiffuseParticlesH = PX_PINNED_MEMORY_ALLOC(int, mContextManager, 1);
		*mNumActiveDiffuseParticlesH = 0;

		PxCudaContext* cudaContext = mContextManager.getCudaContext();
		cudaContext->memsetD32((CUdeviceptr)mNumDiffuseParticlesD, 0, 2);
		mMaxNumDiffuseParticles = maxNumDiffuseParticles;
		mContextManager.releaseContext();
	}

	PxgParticleAndDiffuseBuffer::~PxgParticleAndDiffuseBuffer()
	{
		PX_DEVICE_MEMORY_FREE(mContextManager, mDiffusePositionsLifeTimeD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mDiffuseVelocitiesD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mNumDiffuseParticlesD);
		PX_PINNED_MEMORY_FREE(mContextManager, mNumActiveDiffuseParticlesH);
	}

	void PxgParticleAndDiffuseBuffer::setMaxActiveDiffuseParticles(PxU32 maxActiveDiffuseParticles)
	{
		PX_ASSERT(maxActiveDiffuseParticles <= mMaxNumDiffuseParticles);
		mMaxActiveDiffuseParticles = maxActiveDiffuseParticles;
		mBufferFlags |= PxParticleBufferFlag::eUPDATE_DIFFUSE_PARAM;
	}

	void PxgParticleAndDiffuseBuffer::setDiffuseParticleParams(const PxDiffuseParticleParams& params)
	{
		mParams = params;
		mBufferFlags |= PxParticleBufferFlag::eUPDATE_DIFFUSE_PARAM;
	}

	void PxgParticleAndDiffuseBuffer::copyToHost(CUstream stream)
	{
		PxgParticleBufferBase<PxsParticleAndDiffuseBuffer>::copyToHost(stream);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	PxgParticleClothBuffer::PxgParticleClothBuffer(PxU32 maxNumParticles, PxU32 maxNumVolumes, PxU32 maxNumCloths, 
		PxU32 maxNumTriangles, PxU32 maxNumSprings, PxCudaContextManager& contextManager) :
		PxgParticleBufferBase<PxsParticleClothBuffer>(maxNumParticles, maxNumVolumes, contextManager),
		mRestPositionsD(NULL),
		mTriangleIndicesD(NULL),
		mAccumulatedSpringsPerPartitionsD(NULL),
		mAccumulatedCopiesPerParticlesD(NULL),
		mRemapOutputD(NULL),
		mOrderedSpringsD(NULL),
		mSortedClothStartIndicesD(NULL),
		mClothsD(NULL),
		mRemapPositionsD(NULL),
		mRemapVelocitiesD(NULL),
		mSpringLambdaD(NULL),
		mInflatableLambdaD(NULL),
		mMaxNumCloths(maxNumCloths),
		mMaxNumTriangles(maxNumTriangles),
		mMaxNumSprings(maxNumSprings),
		mNumPartitions(0),
		mMaxSpringsPerPartition(0),
		mNumSprings(0),
		mNumCloths(0),
		mNumTriangles(0),
		mRemapOutputSize(0)
	{
		PX_ASSERT(maxNumParticles > 0 && maxNumTriangles > 0);
		mRestPositionsD = PX_DEVICE_MEMORY_ALLOC(PxVec4, contextManager, maxNumParticles);
		mTriangleIndicesD = PX_DEVICE_MEMORY_ALLOC(PxU32, contextManager, maxNumTriangles * 3);
	}

	PxgParticleClothBuffer::~PxgParticleClothBuffer()
	{
		PX_DEVICE_MEMORY_FREE(mContextManager, mRestPositionsD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mTriangleIndicesD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mAccumulatedSpringsPerPartitionsD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mAccumulatedCopiesPerParticlesD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mRemapOutputD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mOrderedSpringsD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mSortedClothStartIndicesD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mClothsD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mRemapPositionsD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mRemapVelocitiesD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mSpringLambdaD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mInflatableLambdaD);
	}

	void PxgParticleClothBuffer::setNbActiveParticles(PxU32 numActiveParticles)
	{
		PxgParticleBufferBase<PxsParticleClothBuffer>::setNbActiveParticles(numActiveParticles);
		mBufferFlags |= PxParticleBufferFlag::eUPDATE_RESTPOSITION;
	}

	void PxgParticleClothBuffer::setCloths(PxPartitionedParticleCloth& cloths)
	{
		PX_ASSERT(mNumActiveParticles > 0 && mNumActiveParticles <= mMaxNumParticles);

		mContextManager.acquireContext();
		PxCudaContext* cudaContext = mContextManager.getCudaContext();

		if (mNumSprings != cloths.nbSprings || mNumCloths != cloths.nbCloths || mNumPartitions != cloths.nbPartitions || mRemapOutputSize != cloths.remapOutputSize)
		{
			mNumSprings = cloths.nbSprings;
			mNumCloths = cloths.nbCloths;
			mNumPartitions = cloths.nbPartitions;
			mRemapOutputSize = cloths.remapOutputSize;

			PX_DEVICE_MEMORY_FREE(mContextManager, mAccumulatedCopiesPerParticlesD);
			PX_DEVICE_MEMORY_FREE(mContextManager, mAccumulatedSpringsPerPartitionsD);
			PX_DEVICE_MEMORY_FREE(mContextManager, mRemapOutputD);
			PX_DEVICE_MEMORY_FREE(mContextManager, mOrderedSpringsD);
			PX_DEVICE_MEMORY_FREE(mContextManager, mSpringLambdaD);
			PX_DEVICE_MEMORY_FREE(mContextManager, mSortedClothStartIndicesD);
			PX_DEVICE_MEMORY_FREE(mContextManager, mClothsD);
			PX_DEVICE_MEMORY_FREE(mContextManager, mRemapPositionsD);
			PX_DEVICE_MEMORY_FREE(mContextManager, mRemapVelocitiesD);
			PX_DEVICE_MEMORY_FREE(mContextManager, mInflatableLambdaD);

			mAccumulatedCopiesPerParticlesD = PX_DEVICE_MEMORY_ALLOC(PxU32, mContextManager, mNumActiveParticles);
			mAccumulatedSpringsPerPartitionsD = PX_DEVICE_MEMORY_ALLOC(PxU32, mContextManager, mNumPartitions);
			if (mNumSprings > 0)
			{
				mRemapOutputD = PX_DEVICE_MEMORY_ALLOC(PxU32, mContextManager, mNumSprings * 2);
				mOrderedSpringsD = PX_DEVICE_MEMORY_ALLOC(PxParticleSpring, mContextManager, mNumSprings);
				mSpringLambdaD = PX_DEVICE_MEMORY_ALLOC(PxReal, mContextManager, mNumSprings);
			}
			else
			{
				mRemapOutputD = NULL;
				mOrderedSpringsD = NULL;
				mSpringLambdaD = NULL;
			}
			mSortedClothStartIndicesD = PX_DEVICE_MEMORY_ALLOC(PxU32, mContextManager, mNumCloths);
			mClothsD = PX_DEVICE_MEMORY_ALLOC(PxParticleCloth, mContextManager, mNumCloths);
			mRemapPositionsD = PX_DEVICE_MEMORY_ALLOC(PxVec4, mContextManager, mRemapOutputSize);
			mRemapVelocitiesD = PX_DEVICE_MEMORY_ALLOC(PxVec4, mContextManager, mRemapOutputSize);
			mInflatableLambdaD = PX_DEVICE_MEMORY_ALLOC(PxReal, mContextManager, mNumCloths);
		}

		
		mMaxSpringsPerPartition = cloths.maxSpringsPerPartition;

		//cuEventSynchronize((CUevent)waitEvent);
		cudaContext->memcpyHtoD(CUdeviceptr(mAccumulatedCopiesPerParticlesD), cloths.accumulatedCopiesPerParticles, size_t(sizeof(PxU32) * mNumActiveParticles));
		cudaContext->memcpyHtoD(CUdeviceptr(mAccumulatedSpringsPerPartitionsD), cloths.accumulatedSpringsPerPartitions, size_t(sizeof(PxU32) * mNumPartitions));
		cudaContext->memcpyHtoD(CUdeviceptr(mRemapOutputD), cloths.remapOutput, size_t(sizeof(PxU32) * mNumSprings * 2));
		cudaContext->memcpyHtoD(CUdeviceptr(mOrderedSpringsD), cloths.orderedSprings, size_t(sizeof(PxParticleSpring) * mNumSprings));

		cudaContext->memcpyHtoD(CUdeviceptr(mSortedClothStartIndicesD), cloths.sortedClothStartIndices, size_t(sizeof(PxU32) * mNumCloths));
		cudaContext->memcpyHtoD(CUdeviceptr(mClothsD), cloths.cloths, size_t(sizeof(PxParticleCloth) * mNumCloths));

		mBufferFlags |= PxParticleBufferFlag::eUPDATE_CLOTH;

		mContextManager.releaseContext();
	}

	void PxgParticleClothBuffer::copyToHost(CUstream stream)
	{
		PxgParticleBufferBase<PxsParticleClothBuffer>::copyToHost(stream);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	PxgParticleRigidBuffer::PxgParticleRigidBuffer(PxU32 maxNumParticles, PxU32 maxNumVolumes, PxU32 maxNumRigids, PxCudaContextManager& contextManager) :
		PxgParticleBufferBase<PxsParticleRigidBuffer>(maxNumParticles, maxNumVolumes, contextManager),
		mRigidOffsetsD(NULL),
		mRigidCoefficientsD(NULL),
		mRigidLocalPositionsD(NULL),
		mRigidLocalNormalsD(NULL),
		mRigidTranslationsD(NULL),
		mRigidRotationsD(NULL),
		mNumActiveRigids(0),
		mMaxNumRigids(maxNumRigids)
	{
		PX_ASSERT(maxNumRigids > 0 && maxNumParticles > 0);

		mRigidOffsetsD = PX_DEVICE_MEMORY_ALLOC(PxU32, mContextManager, (maxNumRigids + 1));
		mRigidCoefficientsD = PX_DEVICE_MEMORY_ALLOC(PxReal, mContextManager, maxNumRigids);
		mRigidTranslationsD = PX_DEVICE_MEMORY_ALLOC(PxVec4, mContextManager, maxNumRigids);
		mRigidRotationsD = PX_DEVICE_MEMORY_ALLOC(PxVec4, mContextManager, maxNumRigids);

		mRigidLocalPositionsD = PX_DEVICE_MEMORY_ALLOC(PxVec4, mContextManager, maxNumParticles);
		mRigidLocalNormalsD = PX_DEVICE_MEMORY_ALLOC(PxVec4, mContextManager, maxNumParticles);
	}

	PxgParticleRigidBuffer::~PxgParticleRigidBuffer()
	{
		PX_DEVICE_MEMORY_FREE(mContextManager, mRigidOffsetsD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mRigidCoefficientsD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mRigidTranslationsD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mRigidRotationsD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mRigidLocalNormalsD);
		PX_DEVICE_MEMORY_FREE(mContextManager, mRigidLocalPositionsD);
	}

	void PxgParticleRigidBuffer::copyToHost(CUstream stream)
	{
		PxgParticleBufferBase<PxsParticleRigidBuffer>::copyToHost(stream);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	PxgParticleSystemCore::PxgParticleSystemCore(PxgCudaKernelWranglerManager* gpuKernelWrangler, PxCudaContextManager* cudaContextManager,
		PxgHeapMemoryAllocatorManager* heapMemoryManager, PxgSimulationController* simController, PxgGpuContext* gpuContext, const PxU32 maxParticleContacts) :
		PxgNonRigidCore(gpuKernelWrangler, cudaContextManager, heapMemoryManager, simController, gpuContext, maxParticleContacts, 0, PxsHeapStats::eSHARED_PARTICLES),
		mNewParticleSystemPool(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mParticleSystemPool(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mDirtyParamsParticleSystems(PxVirtualAllocator(heapMemoryManager->mMappedMemoryAllocators)),
		mParticleSystemBuffer(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mActiveParticleSystemBuffer(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mTempGridParticleHashBuf(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mTempGridParticleIndexBuf(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mTempGridDiffuseParticleHashBuf(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mTempGridDiffuseParticleIndexBuf(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mTempCellsHistogramBuf(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mTempBlockCellsHistogramBuf(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mTempHistogramCountBuf(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mTempBoundsBuf(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mPrimitiveContactsBuf(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mPrimitiveContactCountBuf(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mPrimitiveContactSortedByParticleBuf(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mPrimitiveContactSortedByRigidBuf(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mPrimitiveConstraintSortedByParticleBuf(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mPrimitiveConstraintSortedByRigidBuf(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mPrimitiveConstraintAppliedParticleForces(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mPrimitiveConstraintAppliedRigidForces(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mDeltaVelParticleBuf(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mDeltaVelRigidBuf(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mTempBlockDeltaVelBuf(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mTempBlockRigidIdBuf(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mParticleRigidConstraints(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mParticleRigidAttachmentIds(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mParticleRigidConstraintCount(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mParticleRigidAttachmentScaleBuffer(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),
		mHasFlipPhase(heapMemoryManager, PxsHeapStats::eSHARED_PARTICLES),	
		mDiffuseParticlesRSDesc(heapMemoryManager->mMappedMemoryAllocators),
		mNbTotalParticleSystems(0),
		mMaxParticles(0),
		mHasNonZeroFluidBoundaryScale(true),
		mMaxParticlesPerBuffer(0),
		mMaxBuffersPerSystem(0),
		mMaxDiffusePerBuffer(0),
		mMaxDiffuseBuffersPerSystem(0),
		mMaxRigidAttachmentsPerSystem(0),
		mHostContactCount(NULL)
	{
		mCudaContextManager->acquireContext();

		int leastPriority, mostPriority;
		cuCtxGetStreamPriorityRange(&leastPriority, &mostPriority);

		mCudaContext->streamCreateWithPriority(&mStream, CU_STREAM_NON_BLOCKING, leastPriority);
		mCudaContext->streamCreateWithPriority(&mFinalizeStream, CU_STREAM_NON_BLOCKING, leastPriority);

		mCudaContext->eventCreate(&mBoundUpdateEvent, CU_EVENT_DISABLE_TIMING);
		mCudaContext->eventCreate(&mSolveParticleEvent, CU_EVENT_DISABLE_TIMING);
		mCudaContext->eventCreate(&mSelfCollisionEvent, CU_EVENT_DISABLE_TIMING);
		mCudaContext->eventCreate(&mFinalizeStartEvent, CU_EVENT_DISABLE_TIMING);
		mCudaContext->eventCreate(&mSolveParticleRigidEvent, CU_EVENT_DISABLE_TIMING);
		mCudaContext->eventCreate(&mSolveRigidParticleEvent, CU_EVENT_DISABLE_TIMING);

#if 1

		mTempCellsHistogramBuf.allocate(1024 * 1024 * 64, PX_FL);
		mTempBlockCellsHistogramBuf.allocate(PxgParticleSystemKernelGridDim::BOUNDCELLUPDATE * sizeof(PxU32), PX_FL);
		mTempHistogramCountBuf.allocate(sizeof(PxU32), PX_FL);

		const PxU32 contactBufSize = mMaxContacts * sizeof(PxgParticlePrimitiveContact);
		const PxU32 constraintBufSize = ((mMaxContacts+31)/32) * sizeof(PxgParticlePrimitiveConstraintBlock);
		const PxU32 appliedForceSize = mMaxContacts * sizeof(float2);

		mPrimitiveContactsBuf.allocate(contactBufSize, PX_FL);
		mPrimitiveContactCountBuf.allocate(sizeof(PxU32), PX_FL);

		mPrimitiveContactSortedByParticleBuf.allocate(contactBufSize, PX_FL);
		mPrimitiveContactSortedByRigidBuf.allocate(contactBufSize, PX_FL);

		mPrimitiveConstraintSortedByParticleBuf.allocate(constraintBufSize, PX_FL);
		mPrimitiveConstraintSortedByRigidBuf.allocate(constraintBufSize, PX_FL);

		mPrimitiveConstraintAppliedParticleForces.allocate(appliedForceSize, PX_FL);
		mPrimitiveConstraintAppliedRigidForces.allocate(appliedForceSize, PX_FL);

		mDeltaVelParticleBuf.allocate(mMaxContacts *sizeof(float4), PX_FL);
		mDeltaVelRigidBuf.allocate(mMaxContacts *sizeof(float4)*2, PX_FL);

		mTempBlockDeltaVelBuf.allocate(sizeof(PxVec4) * 2 * PxgParticleSystemKernelGridDim::ACCUMULATE_DELTA, PX_FL);
		mTempBlockRigidIdBuf.allocate(sizeof(PxU64) * PxgParticleSystemKernelGridDim::ACCUMULATE_DELTA, PX_FL);

		mHostContactCount = PX_PINNED_MEMORY_ALLOC(PxU32, *mCudaContextManager, 1);
		*mHostContactCount = 0;

#endif
		mCudaContextManager->releaseContext();


		mCurrentTMIndex = 0;

	}

	PxgParticleSystemCore::~PxgParticleSystemCore()
	{
		mCudaContextManager->acquireContext();

		//destroy stream
		mCudaContext->streamDestroy(mFinalizeStream);

		mCudaContext->eventDestroy(mBoundUpdateEvent);
		mCudaContext->eventDestroy(mSolveParticleEvent);
		mCudaContext->eventDestroy(mFinalizeStartEvent);
		mCudaContext->eventDestroy(mSolveParticleRigidEvent);
		mCudaContext->eventDestroy(mSolveRigidParticleEvent);

		PX_PINNED_MEMORY_FREE(*mCudaContextManager, mHostContactCount);

		mCudaContextManager->releaseContext();
	}

	void PxgParticleSystemCore::calculateHash(CUdeviceptr particleSystemsd, CUdeviceptr activeParticleSystemsd, const PxU32 numActiveParticleSystems)
	{
		if (mMaxParticles)
		{
			const CUfunction calcHashKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_CALCULATE_HASH);

			PxCudaKernelParam kernelParams[] =
			{
				PX_CUDA_KERNEL_PARAM(particleSystemsd),
				PX_CUDA_KERNEL_PARAM(activeParticleSystemsd)
			};

			{
				const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::UPDATEGRID;
				const PxU32 numBlocks = (mMaxParticles + numThreadsPerBlock - 1) / numThreadsPerBlock;
				CUresult result = mCudaContext->launchKernel(calcHashKernelFunction, numBlocks, numActiveParticleSystems, 1, numThreadsPerBlock, 1, 1, 0, mStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);
				PX_UNUSED(result);

#if PS_GPU_DEBUG
				result = mCudaContext->streamSynchronize(mStream);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU calculateHash kernel fail!\n");
#endif
			}
		}

	}

	void PxgParticleSystemCore::reorderDataAndFindCellStart(PxgParticleSystem* particleSystems, CUdeviceptr particleSystemsd, const PxU32 id, const PxU32 numParticles)
	{
		//PxgParticleSystem* particleSystems = mSimController->getParticleSystems();
		PxgParticleSystem& particleSystem = particleSystems[id];
		CUdeviceptr cellStartd = reinterpret_cast<CUdeviceptr>(particleSystem.mCellStart);
		CUdeviceptr diffuseCellStartd = reinterpret_cast<CUdeviceptr>(particleSystem.mDiffuseCellStart);
		const PxU32 numCells = particleSystem.getNumCells();

		// set all cells to empty
		mCudaContext->memsetD32Async(cellStartd, EMPTY_CELL, numCells, mStream);
		if (particleSystem.mCommonData.mMaxDiffuseParticles > 0)
			mCudaContext->memsetD32Async(diffuseCellStartd, EMPTY_CELL, numCells, mStream);
		
		if (numParticles == 0)
			return;
		//cudaMemset((void*)cellStart, 0xffffffff, numCells * sizeof(PxU32));

		

		
		const CUfunction findCellKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_REORDER_PARTICLE_FIND_CELLSTARTEND);

		PxCudaKernelParam kernelParams[] =
		{
			PX_CUDA_KERNEL_PARAM(particleSystemsd),
			PX_CUDA_KERNEL_PARAM(id)
		};

		{
			const PxU32 numThreadPerBlock = PxgParticleSystemKernelBlockDim::UPDATEGRID;
			const PxU32 numBlocks = (numParticles + numThreadPerBlock - 1) / numThreadPerBlock;
			CUresult result = mCudaContext->launchKernel(findCellKernelFunction, numBlocks, 1, 2, numThreadPerBlock, 1, 1, 0, mStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
			PX_ASSERT(result == CUDA_SUCCESS);
			PX_UNUSED(result);

			/*result = mCudaContext->streamSynchronize(mStream);
			if (result != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU reorderDataAndFindCellStartD kernel fail!\n");*/

#if PS_GPU_DEBUG
			result = mCudaContext->streamSynchronize(mStream);
			if (result != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU reorderDataAndFindCellStartD kernel fail!\n");


			/*CUdeviceptr sortedPosd = reinterpret_cast<CUdeviceptr>(particleSystem.mSortedPostion_Mass);

			PxArray<PxVec4> sortedPos;
			sortedPos.reserve(numParticles);
			sortedPos.forceSize_Unsafe(numParticles);

			PxArray<PxU32> tCellStarts;
			tCellStarts.reserve(numCells);
			tCellStarts.forceSize_Unsafe(numCells);

			PxArray<PxU32> tCellEnds;
			tCellEnds.reserve(numCells);
			tCellEnds.forceSize_Unsafe(numCells);

			PxArray<PxU32> hash;
			PxArray<PxU32> particleIndex;
			hash.reserve(numParticles);
			hash.forceSize_Unsafe(numParticles);

			particleIndex.reserve(numParticles);
			particleIndex.forceSize_Unsafe(numParticles);

			CUdeviceptr hashd = reinterpret_cast<CUdeviceptr>(particleSystem.mGridParticleHash);
			CUdeviceptr particleIndexd = reinterpret_cast<CUdeviceptr>(particleSystem.mGridParticleIndex);
			CUdeviceptr cellEndd = reinterpret_cast<CUdeviceptr>(particleSystem.mCellEnd);

			mCudaContext->memcpyDtoH(hash.begin(), hashd, sizeof(PxU32) * numParticles);
			mCudaContext->memcpyDtoH(particleIndex.begin(), particleIndexd, sizeof(PxU32) * numParticles);

			mCudaContext->memcpyDtoH(tCellStarts.begin(), cellStartd, sizeof(PxU32) * numCells);
			mCudaContext->memcpyDtoH(tCellEnds.begin(), cellEndd, sizeof(PxU32) * numCells);

			
			mCudaContext->memcpyDtoH(sortedPos.begin(), sortedPosd, sizeof(PxVec4) * numParticles);
			mCudaContext->memcpyDtoH(sortedPos.begin(), sortedPosd, sizeof(PxVec4) * numParticles);

			int bob = 0;
			PX_UNUSED(bob);*/

#endif


#if defined(PX_DCU_PORT) && PX_DCU_PORT
		// Host-side validation of the ordinary-particle grid. Keep this opt-in so
		// normal DCU and NVIDIA runs do not pay for synchronous readbacks.
		const char* gridDiag = std::getenv("PX_DCU_PBD_GRID_DIAG");
		if (gridDiag && gridDiag[0] != '0' && particleSystem.mCommonData.mNumParticles > 0)
		{
			const PxU32 ordinaryParticles = particleSystem.mCommonData.mNumParticles;
			const PxU32 gridCellCount = particleSystem.getNumCells();
			const CUresult syncResult = mCudaContext->streamSynchronize(mStream);

			if (syncResult != CUDA_SUCCESS)
			{
				std::printf("[DCU PBD GRID] marker=PX_DCU_PBD_GRID_DIAG_V1 id=%u sync=0 result=%d particles=%u cells=%u\n",
					id, int(syncResult), ordinaryParticles, gridCellCount);
			}
			else
			{
				std::vector<PxU32> hashes(ordinaryParticles);
				std::vector<PxU32> sortedToUnsorted(ordinaryParticles);
				std::vector<PxU32> unsortedToSorted(ordinaryParticles);
				std::vector<PxU32> cellStarts(gridCellCount);
				std::vector<PxU32> cellEnds(gridCellCount);
				std::vector<float4> sortedPositions(ordinaryParticles);
				std::vector<float4> unsortedPositions(ordinaryParticles);

				const PxCUresult hashResult = mCudaContext->memcpyDtoH(hashes.data(),
					reinterpret_cast<CUdeviceptr>(particleSystem.mGridParticleHash), sizeof(PxU32) * ordinaryParticles);
				const PxCUresult sortedToUnsortedResult = mCudaContext->memcpyDtoH(sortedToUnsorted.data(),
					reinterpret_cast<CUdeviceptr>(particleSystem.mSortedToUnsortedMapping), sizeof(PxU32) * ordinaryParticles);
				const PxCUresult unsortedToSortedResult = mCudaContext->memcpyDtoH(unsortedToSorted.data(),
					reinterpret_cast<CUdeviceptr>(particleSystem.mUnsortedToSortedMapping), sizeof(PxU32) * ordinaryParticles);
				const PxCUresult cellStartResult = mCudaContext->memcpyDtoH(cellStarts.data(),
					reinterpret_cast<CUdeviceptr>(particleSystem.mCellStart), sizeof(PxU32) * gridCellCount);
				const PxCUresult cellEndResult = mCudaContext->memcpyDtoH(cellEnds.data(),
					reinterpret_cast<CUdeviceptr>(particleSystem.mCellEnd), sizeof(PxU32) * gridCellCount);
				const PxCUresult sortedPositionResult = mCudaContext->memcpyDtoH(sortedPositions.data(),
					reinterpret_cast<CUdeviceptr>(particleSystem.mSortedPositions_InvMass), sizeof(float4) * ordinaryParticles);
				const PxCUresult unsortedPositionResult = mCudaContext->memcpyDtoH(unsortedPositions.data(),
					reinterpret_cast<CUdeviceptr>(particleSystem.mUnsortedPositions_InvMass), sizeof(float4) * ordinaryParticles);

				if (hashResult != CUDA_SUCCESS || sortedToUnsortedResult != CUDA_SUCCESS ||
					unsortedToSortedResult != CUDA_SUCCESS || cellStartResult != CUDA_SUCCESS ||
					cellEndResult != CUDA_SUCCESS || sortedPositionResult != CUDA_SUCCESS ||
					unsortedPositionResult != CUDA_SUCCESS)
				{
					std::printf("[DCU PBD GRID] marker=PX_DCU_PBD_GRID_DIAG_V1 id=%u sync=1 copy=0 hash=%d s2u=%d u2s=%d cellStart=%d cellEnd=%d sortedPos=%d unsortedPos=%d particles=%u cells=%u\n",
						id, int(hashResult), int(sortedToUnsortedResult), int(unsortedToSortedResult),
						int(cellStartResult), int(cellEndResult), int(sortedPositionResult),
						int(unsortedPositionResult), ordinaryParticles, gridCellCount);
				}
				else
				{
					PxU32 hashOrderErrors = 0;
						PxU32 hashOutOfRange = 0;
						PxU32 sortedToUnsortedOutOfRange = 0;
						PxU32 sortedToUnsortedDuplicates = 0;
						PxU32 sortedToUnsortedMissing = 0;
						PxU32 unsortedToSortedOutOfRange = 0;
						PxU32 reverseMappingErrors = 0;
						PxU32 positionMismatches = 0;
						PxU32 nonFiniteSortedPositions = 0;
						PxU32 nonEmptyCells = 0;
						PxU32 invalidCellRanges = 0;
						PxU32 cellCoverage = 0;
						PxU32 cellHashMismatches = 0;

						std::vector<PxU8> sortedSeen(ordinaryParticles, PxU8(0));
						for (PxU32 i = 0; i < ordinaryParticles; ++i)
						{
							if (hashes[i] >= gridCellCount)
								++hashOutOfRange;
							if (i > 0 && hashes[i] < hashes[i - 1])
								++hashOrderErrors;

							const PxU32 unsortedIndex = sortedToUnsorted[i];
							if (unsortedIndex >= ordinaryParticles)
								++sortedToUnsortedOutOfRange;
							else if (sortedSeen[unsortedIndex])
								++sortedToUnsortedDuplicates;
							else
								sortedSeen[unsortedIndex] = PxU8(1);

							const float4& sortedPosition = sortedPositions[i];
							if (!PxIsFinite(sortedPosition.x) || !PxIsFinite(sortedPosition.y) ||
								!PxIsFinite(sortedPosition.z) || !PxIsFinite(sortedPosition.w))
								++nonFiniteSortedPositions;

							if (unsortedIndex < ordinaryParticles)
							{
								const float4& unsortedPosition = unsortedPositions[unsortedIndex];
								if (sortedPosition.x != unsortedPosition.x || sortedPosition.y != unsortedPosition.y ||
									sortedPosition.z != unsortedPosition.z || sortedPosition.w != unsortedPosition.w)
									++positionMismatches;
							}
						}

						for (PxU32 i = 0; i < ordinaryParticles; ++i)
							if (!sortedSeen[i])
								++sortedToUnsortedMissing;

						for (PxU32 i = 0; i < ordinaryParticles; ++i)
						{
							const PxU32 sortedIndex = unsortedToSorted[i];
							if (sortedIndex >= ordinaryParticles)
								++unsortedToSortedOutOfRange;
							else if (sortedToUnsorted[sortedIndex] != i)
								++reverseMappingErrors;
						}

						for (PxU32 cell = 0; cell < gridCellCount; ++cell)
						{
							const PxU32 start = cellStarts[cell];
							if (start == EMPTY_CELL)
								continue;

							++nonEmptyCells;
							const PxU32 end = cellEnds[cell];
							if (end == EMPTY_CELL || start >= end || end > ordinaryParticles)
							{
								++invalidCellRanges;
								continue;
							}

							cellCoverage += end - start;
							for (PxU32 i = start; i < end; ++i)
								if (hashes[i] != cell)
									++cellHashMismatches;
						}

						const PxU32 firstHash = ordinaryParticles ? hashes[0] : EMPTY_CELL;
						const PxU32 lastHash = ordinaryParticles ? hashes[ordinaryParticles - 1] : EMPTY_CELL;
						const PxU32 firstMapping = ordinaryParticles ? sortedToUnsorted[0] : EMPTY_CELL;
						const PxU32 lastMapping = ordinaryParticles ? sortedToUnsorted[ordinaryParticles - 1] : EMPTY_CELL;
					std::printf("[DCU PBD GRID] marker=PX_DCU_PBD_GRID_DIAG_V1 id=%u sync=1 copy=1 particles=%u cells=%u grid=%ux%ux%u cellWidth=%.9g hashFirst=%u hashLast=%u hashOrderErrors=%u hashOutOfRange=%u s2uFirst=%u s2uLast=%u s2uOutOfRange=%u s2uDuplicates=%u s2uMissing=%u u2sOutOfRange=%u reverseErrors=%u positionMismatches=%u nonFiniteSorted=%u nonEmptyCells=%u invalidCellRanges=%u cellCoverage=%u cellHashMismatches=%u\n",
							id, ordinaryParticles, gridCellCount, particleSystem.mCommonData.mGridSizeX,
							particleSystem.mCommonData.mGridSizeY, particleSystem.mCommonData.mGridSizeZ,
							particleSystem.mCommonData.mGridCellWidth, firstHash, lastHash, hashOrderErrors,
							hashOutOfRange, firstMapping, lastMapping, sortedToUnsortedOutOfRange,
							sortedToUnsortedDuplicates, sortedToUnsortedMissing, unsortedToSortedOutOfRange,
							reverseMappingErrors, positionMismatches, nonFiniteSortedPositions, nonEmptyCells,
							invalidCellRanges, cellCoverage, cellHashMismatches);
				}
			}
		}
#endif

#if defined(PX_DCU_PORT) && PX_DCU_PORT
		dcuPbdPipelineCheckpoint(mCudaContext, mStream, "grid_reorder_complete", 0,
			particleSystem.mCommonData.mNumParticles);
#endif

		}
	}


	void PxgParticleSystemCore::copyUserBufferToUnsortedArray(CUdeviceptr particleSystemsd, CUdeviceptr activeParticleSystemsd, const PxU32 nbActiveParticles, CUstream bpStream)
	{
		const CUfunction preIntegrateKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_UPDATE_UNSORTED_ARRAY);

		#if defined(PX_DCU_PORT) && PX_DCU_PORT
		const PxU32 numThreadsPerBlock = 256;
		#else
		const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::UPDATEBOUND;
		#endif
		const PxU32 numBlocks = (mMaxParticlesPerBuffer + numThreadsPerBlock - 1) / numThreadsPerBlock;

		if(mMaxParticlesPerBuffer)
		{
			PxCudaKernelParam kernelParams[] =
			{
				PX_CUDA_KERNEL_PARAM(particleSystemsd),
				PX_CUDA_KERNEL_PARAM(activeParticleSystemsd)
			};

	
			CUresult result = mCudaContext->launchKernel(preIntegrateKernelFunction, numBlocks, mMaxBuffersPerSystem, nbActiveParticles, numThreadsPerBlock, 1, 1, 0, bpStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
			PX_ASSERT(result == CUDA_SUCCESS);
			PX_UNUSED(result);


#if PS_GPU_DEBUG
			result = mCudaContext->streamSynchronize(bpStream);
			PX_ASSERT(result == CUDA_SUCCESS);
			if (result != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU ps_updateUnsortedArrayLaunch kernel fail!\n");
#endif
		}

	}

	void PxgParticleSystemCore::copyUserDiffuseBufferToUnsortedArray(CUdeviceptr particleSystemsd, CUdeviceptr activeParticleSystemsd,
		const PxU32 nbActiveParticles, CUstream bpStream)
	{
		if (mMaxDiffusePerBuffer > 0)
		{
			const CUfunction copyToDiffuseUnsortedFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_UPDATE_DIFFUSE_UNSORTED_ARRAY);

#if defined(PX_DCU_PORT) && PX_DCU_PORT
			const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::PS_SOLVE;
#else
			const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::UPDATEBOUND;
#endif
			const PxU32 numBlocks = (mMaxDiffusePerBuffer + numThreadsPerBlock - 1) / numThreadsPerBlock;


			{
				PxCudaKernelParam kernelParams[] =
				{
					PX_CUDA_KERNEL_PARAM(particleSystemsd),
					PX_CUDA_KERNEL_PARAM(activeParticleSystemsd)
				};


				CUresult result = mCudaContext->launchKernel(copyToDiffuseUnsortedFunction, numBlocks, mMaxDiffuseBuffersPerSystem, nbActiveParticles, numThreadsPerBlock, 1, 1, 0, bpStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);
				PX_UNUSED(result);


#if PS_GPU_DEBUG
				result = mCudaContext->streamSynchronize(bpStream);
				PX_ASSERT(result == CUDA_SUCCESS);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU ps_updateUnsortedDiffuseArrayLaunch kernel fail!\n");
#endif
			}
		}
	}

	void PxgParticleSystemCore::preIntegrateSystem(CUdeviceptr particleSystemsd, CUdeviceptr activeParticleSystemsd, const PxU32 nbActiveParticles,
		const PxVec3 gravity, const PxReal dt, CUstream bpStream)
	{
	    const bool isTGS = mGpuContext->isTGS();
	    const bool externalForcesEveryTgsIterationEnabled = mGpuContext->isExternalForcesEveryTgsIterationEnabled();

		if (mMaxParticles)
		{
			const CUfunction preIntegrateKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_PREINTEGRATION);

			const PxU32 numBlocks = (mMaxParticles + PxgParticleSystemKernelBlockDim::UPDATEBOUND - 1) / PxgParticleSystemKernelBlockDim::UPDATEBOUND;

			{
				PxCudaKernelParam kernelParams[] =
				{
					PX_CUDA_KERNEL_PARAM(particleSystemsd),
					PX_CUDA_KERNEL_PARAM(activeParticleSystemsd),
					PX_CUDA_KERNEL_PARAM(gravity),
					PX_CUDA_KERNEL_PARAM(dt),
					PX_CUDA_KERNEL_PARAM(isTGS),
					PX_CUDA_KERNEL_PARAM(externalForcesEveryTgsIterationEnabled)
				};

				//one  blockIdx.z to indicate normal particles or diffuse particles
				CUresult result = mCudaContext->launchKernel(preIntegrateKernelFunction, numBlocks, nbActiveParticles, 1, PxgParticleSystemKernelBlockDim::UPDATEBOUND, 1, 1, 0, bpStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);
				PX_UNUSED(result);


#if PS_GPU_DEBUG
				result = mCudaContext->streamSynchronize(bpStream);
				PX_ASSERT(result == CUDA_SUCCESS);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU preIntegrateSystem kernel fail!\n");
#endif
			}
		}
	
	}

	void PxgParticleSystemCore::updateBound(const PxgParticleSystem& sys, PxgParticleSystem* particleSystems,
		PxBounds3* boundArray, PxReal* contactDists, CUstream bpStream)
	{
		const bool isTGS = mGpuContext->isTGS();
		const CUfunction updateBoundFirstKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_UPDATE_BOUND_FRIST);
		const CUfunction updateBoundSecondKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_UPDATE_BOUND_SECOND);

		const PxU32 numThreadsPerWarp = 32;
		const PxU32 numWarpsPerBlock = PxgParticleSystemKernelBlockDim::UPDATEBOUND / numThreadsPerWarp;
		const PxU32 numDiffuseParticles = sys.mCommonData.mMaxDiffuseParticles;
		const PxU32 numBlocks = (PxMax(sys.mCommonData.mNumParticles, numDiffuseParticles) + PxgParticleSystemKernelBlockDim::UPDATEBOUND - 1) / PxgParticleSystemKernelBlockDim::UPDATEBOUND;
		
		mTempBoundsBuf.allocate(sizeof(PxBounds3) * numBlocks, PX_FL);
		CUdeviceptr tempBuf = mTempBoundsBuf.getDevicePtr();
		const PxU32 particleSystemId = sys.mData.mRemapIndex;

		{
			PxCudaKernelParam kernelParams[] =
			{
				PX_CUDA_KERNEL_PARAM(particleSystems),
				PX_CUDA_KERNEL_PARAM(particleSystemId),
				PX_CUDA_KERNEL_PARAM(isTGS),
				PX_CUDA_KERNEL_PARAM(tempBuf)
			};

			CUresult result = mCudaContext->launchKernel(updateBoundFirstKernelFunction, numBlocks, 1, 1, numThreadsPerWarp, numWarpsPerBlock, 1, 0, bpStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
			PX_ASSERT(result == CUDA_SUCCESS);
			PX_UNUSED(result);

			/*result = mCudaContext->streamSynchronize(mStream);
			PX_ASSERT(result == CUDA_SUCCESS);
			if (result != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU updateBound first pass kernel fail!\n");*
				*/

#if PS_GPU_DEBUG
			result = mCudaContext->streamSynchronize(bpStream);
			PX_ASSERT(result == CUDA_SUCCESS);
			if (result != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU updateBound first pass kernel fail!\n");

			////Dma back the bound
			//PxBounds3 bounds;
			//mCudaContext->memcpyDtoH(&bounds, tempBuf, sizeof(PxBounds3));

			//int bob = 0;
			//PX_UNUSED(bob);
#endif
		}
		{
			//elemId is the index to the bound array in BP
			const PxU32 elemId = sys.mData.mElementIndex;
			PxCudaKernelParam kernelParams[] =
			{
				PX_CUDA_KERNEL_PARAM(particleSystems),
				PX_CUDA_KERNEL_PARAM(particleSystemId),
				PX_CUDA_KERNEL_PARAM(tempBuf),
				PX_CUDA_KERNEL_PARAM(numBlocks),
				PX_CUDA_KERNEL_PARAM(boundArray),
				PX_CUDA_KERNEL_PARAM(contactDists),
				PX_CUDA_KERNEL_PARAM(elemId)
			};

			CUresult result = mCudaContext->launchKernel(updateBoundSecondKernelFunction, 1, 1, 1, numThreadsPerWarp, numWarpsPerBlock, 1, 0, bpStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
			PX_ASSERT(result == CUDA_SUCCESS);
			PX_UNUSED(result);

			/*result = mCudaContext->streamSynchronize(mStream);
			PX_ASSERT(result == CUDA_SUCCESS);
			if (result != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU updateBound second pass kernel fail!\n");*/

#if PS_GPU_DEBUG
			result = mCudaContext->streamSynchronize(bpStream);
			PX_ASSERT(result == CUDA_SUCCESS);
			if (result != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU updateBound second pass kernel fail!\n");

			
			//Dma back the bound
			//PxBounds3 bounds[3];
			//mCudaContext->memcpyDtoH(bounds, CUdeviceptr(boundArray ), sizeof(PxBounds3) * 3);

			//int bob = 0;
			//PX_UNUSED(bob);
#endif
		}

	}

	void PxgParticleSystemCore::updateBounds(PxgParticleSystem* particleSystems, PxU32* activeParticleSystems, const PxU32 nbActiveParticleSystems)
	{
		//boundsd  and contactDistd are guaranteed in the GPU if updateBound run on the broad phase stream
		//update bounds need to run on the broad phase stream so the broad phase can kick off after particle update bound
		CUstream bpStream = mGpuContext->mGpuBp->getBpStream();

		
		PxgCudaBuffer& particleSystemBuffer = getParticleSystemBuffer();
		PxgParticleSystem* particleSystemsd = reinterpret_cast<PxgParticleSystem*>(particleSystemBuffer.getDevicePtr());

		PxBounds3* boundsd = reinterpret_cast<PxBounds3*>(mGpuContext->mGpuBp->getBoundsBuffer().getDevicePtr());
		PxReal* contactDistd = reinterpret_cast<PxReal*>(mGpuContext->mGpuBp->getContactDistBuffer().getDevicePtr());
		for (PxU32 i = 0; i < nbActiveParticleSystems; ++i)
		{
			if (particleSystems[activeParticleSystems[i]].mCommonData.mNumParticles)
			{
				updateBound(particleSystems[activeParticleSystems[i]], particleSystemsd, boundsd, contactDistd, bpStream);
			}
		}


		//mCudaContext->streamFlush(bpStream); //Force update ASAP
	}

	PX_FORCE_INLINE PxI32 getNbBits(PxI32 x)
	{
		PxI32 n = 0;
		while (x >= 2)
		{
			++n;
			x /= 2;
		}

		return n;
	}

	void PxgParticleSystemCore::updateGrid(PxgParticleSystem* particleSystems, const PxU32* activeParticleSystems, 
		const PxU32 nbActiveParticleSystems, CUdeviceptr particleSystemsd)
	{
		PX_PROFILE_ZONE("PxgParticleSystemCore::UpdateGrid", 0);
		
		mTempGridParticleHashBuf.allocate(mMaxParticles * sizeof(PxU32), PX_FL);
		mTempGridParticleIndexBuf.allocate(mMaxParticles * sizeof(PxU32), PX_FL);

		//We need 2x rsDesc on the host per-particle system. The reason for this is that, while the sorting occurs synchronously on the 
		//same stream on the device, the host-side buffers could get changed prior to the DMAs having occurred due to device latency
		const PxU32 nbRequired = (2u + nbActiveParticleSystems);
		mRSDesc.resize(nbRequired * 2u);
		//mDiffuseParticlesRSDesc.resize(nbRequired * 2u);

		mRadixCountTotalBuf.allocate(mRadixCountSize * nbRequired, PX_FL);

		for (PxU32 i = 0; i < 2; ++i)
		{
			mRadixSortDescBuf[i].allocate(sizeof(PxgRadixSortBlockDesc)*nbRequired, PX_FL);
		}

		CUdeviceptr activeParticleSystemsd = getActiveParticleSystemBuffer().getDevicePtr();
		calculateHash(particleSystemsd, activeParticleSystemsd, nbActiveParticleSystems);


		for (PxU32 i = 0; i < nbActiveParticleSystems; ++i)
		{
			PxU32 index = activeParticleSystems[i];
			PxgParticleSystem& ps = particleSystems[index];

			if (particleSystems[activeParticleSystems[i]].mCommonData.mNumParticles)
			{
				// Simulation particles
				{
					const PxU32 numSimParticles = ps.mCommonData.mNumParticles;
					
					PxgRadixSortDesc* rsDescs = &mRSDesc[i * 2];

					CUdeviceptr inputKeyd = reinterpret_cast<CUdeviceptr>(ps.mGridParticleHash);
					CUdeviceptr inputRankd = reinterpret_cast<CUdeviceptr>(ps.mSortedToUnsortedMapping);
					CUdeviceptr outputKeyd = mTempGridParticleHashBuf.getDevicePtr();
					CUdeviceptr outputRankd = mTempGridParticleIndexBuf.getDevicePtr();
					PxgRadixSortCore::updateGPURadixSortDesc(mCudaContext, mStream, inputKeyd, inputRankd, outputKeyd, outputRankd, mRadixCountTotalBuf.getDevicePtr(), rsDescs,
						mRadixSortDescBuf[0].getDevicePtr(), mRadixSortDescBuf[1].getDevicePtr(), numSimParticles);

					PxgRadixSortCore::sort(mGpuKernelWranglerManager, mCudaContext, mStream, numSimParticles, mRadixSortDescBuf.begin(), PxgRadixSortCore::getNbBits(PxI32(ps.mCommonData.mGridSizeX*ps.mCommonData.mGridSizeY*ps.mCommonData.mGridSizeZ)), rsDescs);
				}
			}

			const PxU32 numParticles = PxMax(ps.mCommonData.mNumParticles, ps.mCommonData.mMaxDiffuseParticles);
			reorderDataAndFindCellStart(particleSystems, particleSystemsd, index, numParticles);
		}
	}

	void PxgParticleSystemCore::selfCollision(PxgParticleSystem& particleSystem, PxgParticleSystem* particleSystemsd, const PxU32 id, const PxU32 numParticles)
	{
		PX_UNUSED(particleSystem);
		//use the histogram count for the shared index for self collision
		CUdeviceptr sharedIndex = mTempHistogramCountBuf.getDevicePtr();
		mCudaContext->memsetD32Async(sharedIndex, 0, 1, mStream);

		if(!(particleSystem.mData.mFlags & PxParticleFlag::eDISABLE_SELF_COLLISION))
		{
			const CUfunction selfCollisionFirstKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_PARTICLE_SELF_COLLISION);

			PxCudaKernelParam kernelParams[] =
			{
				PX_CUDA_KERNEL_PARAM(particleSystemsd),
				PX_CUDA_KERNEL_PARAM(id)
			};

			{
				const PxU32 numThreadsPerBlock = 64;
				const PxU32 numBlocks = (numParticles + numThreadsPerBlock - 1) / numThreadsPerBlock;
				CUresult result = mCudaContext->launchKernel(selfCollisionFirstKernelFunction, numBlocks, 1, 1, numThreadsPerBlock, 1, 1, 0, mStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);
				PX_UNUSED(result);

				/*result = mCudaContext->streamSynchronize(mStream);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU selfCollisionFirst kernel fail!\n");*/

#if PS_GPU_DEBUG
				result = mCudaContext->streamSynchronize(mStream);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU selfCollisionFirst kernel fail!\n");


				/*PxArray<PxgParticleCollisionHeader> headers;
				headers.reserve(numParticles);
				headers.forceSize_Unsafe(numParticles);
				mCudaContext->memcpyDtoH(headers.begin(), reinterpret_cast<CUdeviceptr>(particleSystem.mCollisionHeaders), sizeof(PxgParticleCollisionHeader) * numParticles);

				int bob;
				PX_UNUSED(bob);*/
#endif
			}
		}

	}


	void PxgParticleSystemCore::resetContactCounts()
	{
		CUdeviceptr totalContactCountsd = getParticleContactCount().getDevicePtr();
		mCudaContext->memsetD32Async(totalContactCountsd, 0, 1, mStream);

		*mHostContactCount = 0;
	}

	//sort contacts based on (1)rigid body index (2)particle id 
	void PxgParticleSystemCore::sortContacts(const PxU32 nbActiveParticleSystems)
	{
	
		CUdeviceptr  totalContactCountsd = mPrimitiveContactCountBuf.getDevicePtr();

		// copy contacts to host for overflow checking.
		mCudaContext->memcpyDtoHAsync(mHostContactCount, totalContactCountsd, sizeof(PxU32), mStream);

		{
			CUfunction clampFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::CLAMP_MAX_VALUE);
			PxCudaKernelParam radixSortKernelParams[] =
			{
				PX_CUDA_KERNEL_PARAM(totalContactCountsd),
				PX_CUDA_KERNEL_PARAM(mMaxContacts)
			};

			CUresult  resultR = mCudaContext->launchKernel(clampFunction, 1, 2, 1, 1, 1, 1, 0, mStream, radixSortKernelParams, sizeof(radixSortKernelParams), 0, PX_FL);
			if (resultR != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU clampMaxValue fail to launch kernel!!\n");

#if defined(PX_DCU_PORT) && PX_DCU_PORT
			dcuPbdPrimitiveSortCheckpoint(mCudaContext, mStream, "contact_clamp_complete",
				totalContactCountsd, nbActiveParticleSystems);
#endif

#if PS_GPU_DEBUG
			resultR = mCudaContext->streamSynchronize(mStream);
			if (resultR != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU clampMaxValue fail!!\n");
#endif

		}

		{
			//sort particle-primitives contact based on rigid body index(lower 32 bit of the node index, which get fill in contact gen)
			CUdeviceptr inputKeyd = mTempContactByRigidBitBuf.getDevicePtr();
			CUdeviceptr inputRankd = mContactRemapSortedByRigidBuf.getDevicePtr();

			CUdeviceptr outputKeyd = mTempContactBuf.getDevicePtr();
			CUdeviceptr outputRankd = mTempContactRemapBuf.getDevicePtr();

			updateGPURadixSortBlockDesc(mStream, inputKeyd, inputRankd, outputKeyd, outputRankd, mRadixCountTotalBuf.getDevicePtr(), 
				totalContactCountsd, &mRSDesc[2 * nbActiveParticleSystems], mRadixSortDescBuf[0].getDevicePtr(), mRadixSortDescBuf[1].getDevicePtr());

			//sort particle-primitives contact based on particle index(lower 32 bit is the particle index, which get fill in contact gen)
			CUdeviceptr inputKeyd2 = mTempContactByParticleBitBuf.getDevicePtr();
			CUdeviceptr inputRankd2 = mContactRemapSortedByParticleBuf.getDevicePtr();
			//reuse the temp buffer
			CUdeviceptr outputKeyd2 = mTempContactBuf2.getDevicePtr();
			CUdeviceptr outputRankd2 = mTempContactRemapBuf2.getDevicePtr();

			updateGPURadixSortBlockDesc(mStream, inputKeyd2, inputRankd2, outputKeyd2, outputRankd2, mRadixCountTotalBuf.getDevicePtr() + mRadixCountSize,
				totalContactCountsd, &mRSDesc[2 * nbActiveParticleSystems + 2], mRadixSortDescBuf[0].getDevicePtr() + sizeof(PxgRadixSortBlockDesc),
				mRadixSortDescBuf[1].getDevicePtr() + sizeof(PxgRadixSortBlockDesc));

#if defined(PX_DCU_PORT) && PX_DCU_PORT
			dcuPbdPrimitiveSortCheckpoint(mCudaContext, mStream, "radix_descriptors_complete",
				totalContactCountsd, nbActiveParticleSystems);
			DcuPbdPrimitiveSortReference primitiveSortReference;
			if (!dcuPbdCapturePrimitiveSortReference(mCudaContext, mStream, totalContactCountsd,
				inputKeyd, inputRankd, inputKeyd2, inputRankd2, mContactByRigidBuf.getDevicePtr(),
				mContactSortedByParticleBuf.getDevicePtr(), mMaxContacts, primitiveSortReference))
				std::abort();
#endif

			PxgCudaBuffer* radixSortDescBuf = mRadixSortDescBuf.begin();
			CUdeviceptr contactByRigidd = mContactByRigidBuf.getDevicePtr();
			CUdeviceptr contactByParticled = mContactSortedByParticleBuf.getDevicePtr();

			#if defined(PX_DCU_PORT) && PX_DCU_PORT
			const bool useChunkedPrimitiveRadix = dcuPbdUseChunkedPrimitiveRadix();
			const bool useInlinePrimitiveRadix = useChunkedPrimitiveRadix && dcuPbdUseInlinePrimitiveRadix();
			const bool useFused64PrimitiveRadix = useInlinePrimitiveRadix && dcuPbdUseFused64PrimitiveRadix();
			dcuPbdPrintPrimitiveRadixModeOnce(useChunkedPrimitiveRadix, useInlinePrimitiveRadix, useFused64PrimitiveRadix);
			CUfunction pbdPrimitiveRadixFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(
				useFused64PrimitiveRadix ? PxgKernelIds::RS_PBD_PRIMITIVE_CHUNKED_COUNTING_64_INLINE_DCU_V63
					: useInlinePrimitiveRadix ? PxgKernelIds::RS_PBD_PRIMITIVE_CHUNKED_COUNTING_32_INLINE_DCU_V7
					: useChunkedPrimitiveRadix ? PxgKernelIds::RS_PBD_PRIMITIVE_CHUNKED_COUNTING_32_DCU_V5
					: PxgKernelIds::RS_PBD_PRIMITIVE_SERIAL_COUNTING_PASS_DCU_V4);
			#else
			CUfunction radixFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::RS_MULTIBLOCK);
			CUfunction calculateRanksFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::RS_CALCULATERANKS_MULTIBLOCK);
			#endif

			{
				#if defined(PX_DCU_PORT) && PX_DCU_PORT
				if(useChunkedPrimitiveRadix)
				{
					CUdeviceptr rsDesc0 = radixSortDescBuf[0].getDevicePtr();
					CUdeviceptr rsDesc1 = radixSortDescBuf[1].getDevicePtr();
					if(useFused64PrimitiveRadix)
					{
						PxCudaKernelParam radixSortKernelParams[] =
						{
							PX_CUDA_KERNEL_PARAM(rsDesc0),
							PX_CUDA_KERNEL_PARAM(rsDesc1),
							PX_CUDA_KERNEL_PARAM(contactByRigidd),
							PX_CUDA_KERNEL_PARAM(contactByParticled)
						};
						const CUresult resultR = mCudaContext->launchKernel(pbdPrimitiveRadixFunction, 1, 2, 1, 32, 1, 1, 0,
							mStream, radixSortKernelParams, sizeof(radixSortKernelParams), 0, PX_FL);
						if(resultR != CUDA_SUCCESS)
							PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU fused64 sortParticleContacts fail to launch kernel!!\n");
					}
					else
					{
						PxCudaKernelParam radixSortKernelParams[] =
						{
							PX_CUDA_KERNEL_PARAM(rsDesc0),
							PX_CUDA_KERNEL_PARAM(rsDesc1)
						};
						const CUresult resultR = mCudaContext->launchKernel(pbdPrimitiveRadixFunction, 1, 2, 1, 32, 1, 1, 0,
							mStream, radixSortKernelParams, sizeof(radixSortKernelParams), 0, PX_FL);
						if(resultR != CUDA_SUCCESS)
							PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU chunked sortParticleContacts fail to launch kernel!!\n");
					}
				}
				else
				#endif
				{
				PxU32 startBit = 0;
				const PxU32 numPass = 8;

				for (PxU32 i = 0; i < numPass; ++i)
				{
					const PxU32 descIndex = i & 1;

					CUdeviceptr rsDesc = radixSortDescBuf[descIndex].getDevicePtr();

					PxCudaKernelParam radixSortKernelParams[] =
					{
						PX_CUDA_KERNEL_PARAM(rsDesc),
						PX_CUDA_KERNEL_PARAM(startBit)
					};

					#if defined(PX_DCU_PORT) && PX_DCU_PORT
					CUresult resultR = mCudaContext->launchKernel(pbdPrimitiveRadixFunction, 1, 2, 1, 1, 1, 1, 0,
						mStream, radixSortKernelParams, sizeof(radixSortKernelParams), 0, PX_FL);
					#else
					CUresult resultR = mCudaContext->launchKernel(radixFunction, PxgRadixSortKernelGridDim::RADIX_SORT, 2, 1,
						PxgRadixSortKernelBlockDim::RADIX_SORT, 1, 1, 0, mStream, radixSortKernelParams,
						sizeof(radixSortKernelParams), 0, PX_FL);
					if (resultR == CUDA_SUCCESS)
						resultR = mCudaContext->launchKernel(calculateRanksFunction, PxgRadixSortKernelGridDim::RADIX_SORT, 2, 1,
							PxgRadixSortKernelBlockDim::RADIX_SORT, 1, 1, 0, mStream, radixSortKernelParams,
							sizeof(radixSortKernelParams), 0, PX_FL);
					#endif
					if (resultR != CUDA_SUCCESS)
						PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU sortParticleContacts fail to launch kernel!!\n");

					startBit += 4;
				}
				}

#if defined(PX_DCU_PORT) && PX_DCU_PORT
				dcuPbdPrimitiveSortCheckpoint(mCudaContext, mStream,
					useFused64PrimitiveRadix ? "radix_fused64_complete" : "radix_low32_complete",
					totalContactCountsd, nbActiveParticleSystems);
				if (!useFused64PrimitiveRadix && !dcuPbdValidatePrimitiveSortOutput(mCudaContext, mStream, totalContactCountsd,
					inputKeyd, inputRankd, inputKeyd2, inputRankd2, primitiveSortReference))
					std::abort();
#endif

				/*CUresult result = mCudaContext->streamSynchronize(mStream);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU sortParticleContacts fail!!\n");*/

#if PS_GPU_DEBUG
				CUresult result = mCudaContext->streamSynchronize(mStream);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU sortParticleContacts fail!!\n");
#endif
			}

			#if defined(PX_DCU_PORT) && PX_DCU_PORT
			if(!useFused64PrimitiveRadix)
			#endif
			{
				//copy the higher 32 bit to the temp contact rigid index buffer
				CUdeviceptr tempContactByRigidd = mTempContactByRigidBitBuf.getDevicePtr();
				CUdeviceptr rankd = mContactRemapSortedByRigidBuf.getDevicePtr();

				//copy the higher 32 bit to the temp contact particle buffer
				CUdeviceptr tempContactByParticled = mTempContactByParticleBitBuf.getDevicePtr();
				CUdeviceptr rankParticled = mContactRemapSortedByParticleBuf.getDevicePtr();

				CUfunction copyFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::RS_DOUBLE_COPY_HIGH_32BITS2);


				PxCudaKernelParam copyKernelParams[] =
				{
					PX_CUDA_KERNEL_PARAM(contactByRigidd),
					PX_CUDA_KERNEL_PARAM(tempContactByRigidd),
					PX_CUDA_KERNEL_PARAM(rankd),
					PX_CUDA_KERNEL_PARAM(contactByParticled),
					PX_CUDA_KERNEL_PARAM(tempContactByParticled),
					PX_CUDA_KERNEL_PARAM(rankParticled),
					PX_CUDA_KERNEL_PARAM(totalContactCountsd)
				};

				#if defined(PX_DCU_PORT) && PX_DCU_PORT
				const PxU32 copyBlockSize = 256;
				#else
				const PxU32 copyBlockSize = PxgRadixSortKernelBlockDim::RADIX_SORT;
				#endif
				CUresult resultR = mCudaContext->launchKernel(copyFunction, 32, 1, 1, copyBlockSize, 1, 1, 0, mStream, copyKernelParams, sizeof(copyKernelParams), 0, PX_FL);
				if (resultR != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU radixSortCopyBits fail to launch kernel!!\n");

#if defined(PX_DCU_PORT) && PX_DCU_PORT
				dcuPbdPrimitiveSortCheckpoint(mCudaContext, mStream, "radix_high32_copy_complete",
					totalContactCountsd, nbActiveParticleSystems);
#endif

#if PS_GPU_DEBUG
				resultR = mCudaContext->streamSynchronize(mStream);
				if (resultR != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU radixSortCopyBits fail!!\n");
#endif
			}

			#if defined(PX_DCU_PORT) && PX_DCU_PORT
			if(!useFused64PrimitiveRadix)
			#endif
			{
				#if defined(PX_DCU_PORT) && PX_DCU_PORT
				if(useChunkedPrimitiveRadix)
				{
					CUdeviceptr rsDesc0 = radixSortDescBuf[0].getDevicePtr();
					CUdeviceptr rsDesc1 = radixSortDescBuf[1].getDevicePtr();
					PxCudaKernelParam radixSortKernelParams[] =
					{
						PX_CUDA_KERNEL_PARAM(rsDesc0),
						PX_CUDA_KERNEL_PARAM(rsDesc1)
					};
					const CUresult resultR = mCudaContext->launchKernel(pbdPrimitiveRadixFunction, 1, 2, 1, 32, 1, 1, 0,
						mStream, radixSortKernelParams, sizeof(radixSortKernelParams), 0, PX_FL);
					if(resultR != CUDA_SUCCESS)
						PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU chunked sortParticleContacts fail to launch kernel!!\n");
				}
				else
				#endif
				{
				//sort tempContactByRidid again
				PxU32 startBit = 0;
				const PxU32 numPass = 8;

				for (PxU32 i = 0; i < numPass; ++i)
				{
					const PxU32 descIndex = i & 1;

					CUdeviceptr rsDesc = radixSortDescBuf[descIndex].getDevicePtr();

					PxCudaKernelParam radixSortKernelParams[] =
					{
						PX_CUDA_KERNEL_PARAM(rsDesc),
						PX_CUDA_KERNEL_PARAM(startBit)
					};

					#if defined(PX_DCU_PORT) && PX_DCU_PORT
					CUresult resultR = mCudaContext->launchKernel(pbdPrimitiveRadixFunction, 1, 2, 1, 1, 1, 1, 0,
						mStream, radixSortKernelParams, sizeof(radixSortKernelParams), 0, PX_FL);
					#else
					CUresult resultR = mCudaContext->launchKernel(radixFunction, PxgRadixSortKernelGridDim::RADIX_SORT, 2, 1,
						PxgRadixSortKernelBlockDim::RADIX_SORT, 1, 1, 0, mStream, radixSortKernelParams,
						sizeof(radixSortKernelParams), 0, PX_FL);
					if (resultR == CUDA_SUCCESS)
						resultR = mCudaContext->launchKernel(calculateRanksFunction, PxgRadixSortKernelGridDim::RADIX_SORT, 2, 1,
							PxgRadixSortKernelBlockDim::RADIX_SORT, 1, 1, 0, mStream, radixSortKernelParams,
							sizeof(radixSortKernelParams), 0, PX_FL);
					#endif
					if (resultR != CUDA_SUCCESS)
						PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU sortParticleContacts fail to launch kernel!!\n");

					startBit += 4;
				}
				}

#if defined(PX_DCU_PORT) && PX_DCU_PORT
				dcuPbdPrimitiveSortCheckpoint(mCudaContext, mStream, "radix_high32_complete",
					totalContactCountsd, nbActiveParticleSystems);
#endif
			}

#if defined(PX_DCU_PORT) && PX_DCU_PORT
			if (!dcuPbdValidatePrimitiveSortOutput64(mCudaContext, mStream, totalContactCountsd,
				mContactRemapSortedByRigidBuf.getDevicePtr(), mContactRemapSortedByParticleBuf.getDevicePtr(),
				contactByRigidd, contactByParticled, primitiveSortReference))
				std::abort();
#endif

			{
				//copy the original rigidId to the sorted buffer based on mContactRemapSortedByRigidBuf
				CUdeviceptr outContactByRigidd = mContactSortedByRigidBuf.getDevicePtr();
				CUdeviceptr rankd = mContactRemapSortedByRigidBuf.getDevicePtr();

				CUdeviceptr outContactByParticled = mContactSortedByParticleBuf.getDevicePtr();
				CUdeviceptr rankParticled = mContactRemapSortedByParticleBuf.getDevicePtr();

				CUfunction copyFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::RS_DOUBLE_COPY_VALUE);

				PxCudaKernelParam copyKernelParams[] =
				{
					PX_CUDA_KERNEL_PARAM(contactByRigidd),
					PX_CUDA_KERNEL_PARAM(outContactByRigidd),
					PX_CUDA_KERNEL_PARAM(rankd),
					PX_CUDA_KERNEL_PARAM(contactByParticled),
					PX_CUDA_KERNEL_PARAM(outContactByParticled),
					PX_CUDA_KERNEL_PARAM(rankParticled),
					PX_CUDA_KERNEL_PARAM(totalContactCountsd)
				};

				#if defined(PX_DCU_PORT) && PX_DCU_PORT
				const PxU32 copyBlockSize = 256;
				#else
				const PxU32 copyBlockSize = PxgRadixSortKernelBlockDim::RADIX_SORT;
				#endif
				CUresult resultR = mCudaContext->launchKernel(copyFunction, 32, 1, 1, copyBlockSize, 1, 1, 0, mStream, copyKernelParams, sizeof(copyKernelParams), 0, PX_FL);
				if (resultR != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU radixSortCopy fail to launch kernel!!\n");

#if defined(PX_DCU_PORT) && PX_DCU_PORT
				dcuPbdPrimitiveSortCheckpoint(mCudaContext, mStream, "radix_sorted_value_copy_complete",
					totalContactCountsd, nbActiveParticleSystems);
#endif

#if PS_GPU_DEBUG
				resultR = mCudaContext->streamSynchronize(mStream);
				if (resultR != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU radixSortCopy fail!!\n");
#endif
			}
		}

		{
			//create sorted contacts based on mContactRemapSortedByRigidBuf and mContactRemapSortedByParticleBuf
			CUdeviceptr remapByRigidd = mContactRemapSortedByRigidBuf.getDevicePtr();
			CUdeviceptr remapByParticled = mContactRemapSortedByParticleBuf.getDevicePtr();

			CUdeviceptr sortedContactsByRigidd = mPrimitiveContactSortedByRigidBuf.getDevicePtr();
			CUdeviceptr sortedContactsByParticled = mPrimitiveContactSortedByParticleBuf.getDevicePtr();
			
			CUdeviceptr contactsd = mPrimitiveContactsBuf.getDevicePtr();

			CUfunction reorderContactsFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_REORDER_PRIMITIVE_CONTACTS);

			PxCudaKernelParam kernelParams[] =
			{
				PX_CUDA_KERNEL_PARAM(contactsd),
				PX_CUDA_KERNEL_PARAM(totalContactCountsd),
				PX_CUDA_KERNEL_PARAM(remapByRigidd),
				PX_CUDA_KERNEL_PARAM(remapByParticled),
				PX_CUDA_KERNEL_PARAM(sortedContactsByRigidd),
				PX_CUDA_KERNEL_PARAM(sortedContactsByParticled)
			};

			#if defined(PX_DCU_PORT) && PX_DCU_PORT
			const PxU32 reorderBlockSize = 256;
			#else
			const PxU32 reorderBlockSize = PxgParticleSystemKernelBlockDim::BOUNDCELLUPDATE;
			#endif
			CUresult  resultR = mCudaContext->launchKernel(reorderContactsFunction, PxgParticleSystemKernelGridDim::BOUNDCELLUPDATE, 1, 1, reorderBlockSize, 1, 1, 0, mStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
			if (resultR != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU sortParticles fail to launch kernel!!\n");

#if defined(PX_DCU_PORT) && PX_DCU_PORT
			dcuPbdPrimitiveSortCheckpoint(mCudaContext, mStream, "primitive_contact_reorder_complete",
				totalContactCountsd, nbActiveParticleSystems);
#endif

#if PS_GPU_DEBUG
			CUresult result = mCudaContext->streamSynchronize(mStream);
			if (result != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU sortParticles fail!!\n");

			//PxU32 numContacts;
			//mCudaContext->memcpyDtoH(&numContacts, totalContactCountsd, sizeof(PxU32));
			//

			/*PxArray<PxgParticlePrimitiveContact> originalContacts;
			originalContacts.reserve(numContacts);
			originalContacts.forceSize_Unsafe(numContacts);
			mCudaContext->memcpyDtoH(originalContacts.begin(), contactsd, sizeof(PxgParticlePrimitiveContact) * numContacts);

			PxArray<PxgParticlePrimitiveContact> rigidContacts;
			rigidContacts.reserve(numContacts);
			rigidContacts.forceSize_Unsafe(numContacts);
			mCudaContext->memcpyDtoH(rigidContacts.begin(), sortedContactsByRigidd, sizeof(PxgParticlePrimitiveContact) * numContacts);

			PxArray<PxgParticlePrimitiveContact> particleContacts;
			particleContacts.reserve(numContacts);
			particleContacts.forceSize_Unsafe(numContacts);
			mCudaContext->memcpyDtoH(particleContacts.begin(), sortedContactsByParticled, sizeof(PxgParticlePrimitiveContact) * numContacts);

			int bob = 0;
			PX_UNUSED(bob);*/
#endif
		}

#if defined(PX_DCU_PORT) && PX_DCU_PORT
		dcuPbdPipelineCheckpoint(mCudaContext, mStream, "primitive_contact_sort_complete",
			totalContactCountsd, nbActiveParticleSystems);
#endif


		//KS - force work to start running!
		//mCudaContext->streamFlush(mStream);

		//At this point, we need to sync the self-stream with the stream
		//synchronizeStreams(mCudaContext, mSelfCollisionStream, mStream, mSelfCollisionEvent);

		/*CUresult result = mCudaContext->streamSynchronize(mStream);
		if (result != CUDA_SUCCESS)
			PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU sortParticles fail!!\n");*/
	}

	void PxgParticleSystemCore::solveOneWayCollision(CUdeviceptr particleSystemsd, CUdeviceptr activeParticleSystemsd, 
		const PxU32 nbActiveParticleSystems, const PxReal dt, const PxReal biasCoefficient, const bool isVelocityIteration)
	{

		const PxU32 maxParticles = getMaxParticles();

		if (maxParticles > 0)
		{
			const PxReal invDt = 1.0f / dt;

			const CUfunction solveSelfCollisionKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_SOLVE_ONEWAY_CONTACTS);

			PxCudaKernelParam kernelParams[] =
			{
				PX_CUDA_KERNEL_PARAM(particleSystemsd),
				PX_CUDA_KERNEL_PARAM(activeParticleSystemsd),
				PX_CUDA_KERNEL_PARAM(invDt),
				PX_CUDA_KERNEL_PARAM(dt),
				PX_CUDA_KERNEL_PARAM(biasCoefficient),
				PX_CUDA_KERNEL_PARAM(isVelocityIteration)
			};

			{
				const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::PS_SOLVE;
				const PxU32 numBlocks = (maxParticles + numThreadsPerBlock - 1) / numThreadsPerBlock;
				CUresult result = mCudaContext->launchKernel(solveSelfCollisionKernelFunction, numBlocks, nbActiveParticleSystems, 1, numThreadsPerBlock, 1, 1, 0, mStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);
				PX_UNUSED(result);

#if PS_GPU_DEBUG
				result = mCudaContext->streamSynchronize(mStream);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU solveOneWayCollision kernel fail!\n");
#endif
			}

			//updateSortedVelocity(particleSystemsd, activeParticleSystemsd, nbActiveParticleSystems, dt);

		}
	}

	//void PxgParticleSystemCore::updateParticles(const PxReal dt)
	//{
	//	updateSortedVelocity(getParticleSystemBuffer().getDevicePtr(),
	//		getActiveParticleSystemBuffer().getDevicePtr(), mSimController->getBodySimManager().mActivePBDParticleSystems.size(), dt);
	//}


//	void PxgParticleSystemCore::solveSprings(CUdeviceptr particleSystemsd, CUdeviceptr activeParticleSystemsd,
//		const PxU32 nbActiveParticleSystems, const PxReal dt, bool isTGS)
//	{
//		PxgSimulationCore* simCore = mSimController->getSimulationCore();
//		const PxU32 maxSprings = simCore->getMaxParticleSprings();
//
//		if (maxSprings > 0)
//		{
//
//			//update duplicate verts
//			{
//				//each block has 1024 threads
//				const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::UPDATEBOUND;
//				const PxU32 numBlocks = (maxSprings * 2 + numThreadsPerBlock - 1) / numThreadsPerBlock;
//
//				PxCudaKernelParam kernelParams[] =
//				{
//					PX_CUDA_KERNEL_PARAM(particleSystemsd),
//					PX_CUDA_KERNEL_PARAM(activeParticleSystemsd)
//				};
//
//				const CUfunction updateRemapKernel = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_UPDATE_REMAP_VERTS);
//
//				CUresult result = mCudaContext->launchKernel(updateRemapKernel, numBlocks, nbActiveParticleSystems, 1, numThreadsPerBlock, 1, 1, 0, mStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
//				PX_ASSERT(result == CUDA_SUCCESS);
//				PX_UNUSED(result);
//
//
//#if PS_GPU_DEBUG
//				{
//					result = mCudaContext->streamSynchronize(mStream);
//					if (result != CUDA_SUCCESS)
//						PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU ps_updateRemapVertsLaunch kernel fail!\n");
//					PX_ASSERT(result == CUDA_SUCCESS);
//				}
//#endif
//			}
//
//
//			const PxReal invDt = 1.f / dt;
//
//			{
//
//				const PxU32 maxPartitions = simCore->getMaxSpringPartitions();
//				const PxU32 maxSpringsPerPartitions = simCore->getMaxSpringsPerPartition();
//				const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::ACCUMULATE_DELTA;
//				const PxU32 numBlocks = (maxSpringsPerPartitions + numThreadsPerBlock - 1) / numThreadsPerBlock;
//
//				for (PxU32 i = 0; i < maxPartitions; ++i)
//				{
//
//					PxCudaKernelParam kernelParams[] =
//					{
//						PX_CUDA_KERNEL_PARAM(particleSystemsd),
//						PX_CUDA_KERNEL_PARAM(activeParticleSystemsd),
//						PX_CUDA_KERNEL_PARAM(nbActiveParticleSystems),
//						PX_CUDA_KERNEL_PARAM(invDt),
//						PX_CUDA_KERNEL_PARAM(i),
//						PX_CUDA_KERNEL_PARAM(isTGS)
//					};
//
//					const CUfunction solveSpringKernel = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_SOLVE_SPRINGS);
//
//					CUresult result = mCudaContext->launchKernel(solveSpringKernel, numBlocks, nbActiveParticleSystems, 1, numThreadsPerBlock, 1, 1, 0, mStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
//					PX_ASSERT(result == CUDA_SUCCESS);
//					PX_UNUSED(result);
//
//
//#if PS_GPU_DEBUG
//					{
//						result = mCudaContext->streamSynchronize(mStream);
//						if (result != CUDA_SUCCESS)
//							PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU ps_solveSpringsPGSLaunch kernel fail!\n");
//						PX_ASSERT(result == CUDA_SUCCESS);
//					}
//#endif
//				}
//			}
//
//			
//			//compute average verts and update sorted positions
//			{
//	
//				//each block has 1024 threads
//				const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::UPDATEBOUND;
//				const PxU32 numBlocks = (maxSprings * 2 + numThreadsPerBlock - 1) / numThreadsPerBlock;
//
//				PxCudaKernelParam kernelParams[] =
//				{
//					PX_CUDA_KERNEL_PARAM(particleSystemsd),
//					PX_CUDA_KERNEL_PARAM(activeParticleSystemsd),
//					PX_CUDA_KERNEL_PARAM(invDt)
//				};
//
//				const CUfunction computeAverageKernel = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_AVERAGEVERTS);
//
//				CUresult result = mCudaContext->launchKernel(computeAverageKernel, numBlocks, nbActiveParticleSystems, 1, numThreadsPerBlock, 1, 1, 0, mStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
//				PX_ASSERT(result == CUDA_SUCCESS);
//				PX_UNUSED(result);
//
//
//#if PS_GPU_DEBUG
//				{
//					result = mCudaContext->streamSynchronize(mStream);
//					if (result != CUDA_SUCCESS)
//						PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU ps_averageVertsLaunch fail!\n");
//					PX_ASSERT(result == CUDA_SUCCESS);
//				}
//#endif
//			}
//
//			//update sorted positions and sorted velocities
//			updateSortedVelocity(particleSystemsd, activeParticleSystemsd, nbActiveParticleSystems, dt);
//
//		}
//
//	}
//

	void PxgParticleSystemCore::applyDeltas(CUdeviceptr particleSystemd, CUdeviceptr activeParticleSystemd, const PxU32 nbActiveParticleSystems, const PxReal dt, CUstream stream)
	{		
		const PxU32 maxParticles = getMaxParticles();
		if (maxParticles > 0)
		{
			const PxU32 numThreadsPerBlock = 64;
			const PxU32 numBlocks = (maxParticles + numThreadsPerBlock - 1) / numThreadsPerBlock;

			{

				const CUfunction applyDeltaKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_APPLY_DELTAS);

				PxCudaKernelParam kernelParams[] =
				{
					PX_CUDA_KERNEL_PARAM(particleSystemd),
					PX_CUDA_KERNEL_PARAM(activeParticleSystemd),
					PX_CUDA_KERNEL_PARAM(dt)
				};

				CUresult result = mCudaContext->launchKernel(applyDeltaKernelFunction, numBlocks, nbActiveParticleSystems, 1, numThreadsPerBlock, 1, 1, 0, stream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);
				PX_UNUSED(result);

#if CLOTH_GPU_DEBUG
				result = mCudaContext->streamSynchronize(mStream);
				PX_ASSERT(result == CUDA_SUCCESS);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU cloth_applyExternalDeltasLaunch first pass kernel fail!\n");
#endif
			}
		}
	}

	void PxgParticleSystemCore::updateSortedVelocity(CUdeviceptr particleSystemsd, CUdeviceptr activeParticleSystemsd,
		const PxU32 nbActiveParticleSystems, const PxReal dt, const bool skipNewPositionAdjustment)
	{

		if (nbActiveParticleSystems == 0)
			return;

		
		const PxU32 maxParticles = getMaxParticles();

		if (maxParticles > 0)
		{

			//each block has 1024 threads
			const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::PS_SOLVE;
			const PxU32 numBlocks = (maxParticles + numThreadsPerBlock - 1) / numThreadsPerBlock;

			const CUfunction updateVelocityKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_UPDATE_PARTICLE);

			PxReal invDt = 1.f/dt;

			PxCudaKernelParam kernelParams[] =
			{
				PX_CUDA_KERNEL_PARAM(particleSystemsd),
				PX_CUDA_KERNEL_PARAM(activeParticleSystemsd),
				PX_CUDA_KERNEL_PARAM(invDt),
				PX_CUDA_KERNEL_PARAM(skipNewPositionAdjustment)
			};

			{
				CUresult result = mCudaContext->launchKernel(updateVelocityKernelFunction, numBlocks, nbActiveParticleSystems, 1, numThreadsPerBlock, 1, 1, 0, mStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);
				PX_UNUSED(result);

#if PS_GPU_DEBUG
				result = mCudaContext->streamSynchronize(mStream);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU ps_updateParticleIntermittentLaunch kernel fail!\n");

				int bob = 0;
				PX_UNUSED(bob);
#endif
			}
		}
	}

	void PxgParticleSystemCore::prepPrimitiveConstraint(CUdeviceptr prePrepDescd, CUdeviceptr prepDescd, CUdeviceptr sharedDescd,
		const PxReal dt, bool isTGS, CUstream solverStream)
	{
		
		CUdeviceptr particleSystemd = getParticleSystemBuffer().getDevicePtr();

		CUdeviceptr totalContactCountsd = getParticleContactCount().getDevicePtr();

		//prepare primitive constraints sorted by rigid id
		{
			CUdeviceptr contactsd = mPrimitiveContactSortedByRigidBuf.getDevicePtr();
			CUdeviceptr constraintsd = mPrimitiveConstraintSortedByRigidBuf.getDevicePtr();
			CUdeviceptr appliedForced = mPrimitiveConstraintAppliedRigidForces.getDevicePtr();

			CUdeviceptr deltaVBuff = mDeltaVelRigidBuf.getDevicePtr();

			const CUfunction prepPrimitiveCollisionKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_CONTACT_PREPARE);

			PxCudaKernelParam kernelParams[] =
			{
				PX_CUDA_KERNEL_PARAM(particleSystemd),
				PX_CUDA_KERNEL_PARAM(contactsd),
				PX_CUDA_KERNEL_PARAM(totalContactCountsd),
				PX_CUDA_KERNEL_PARAM(constraintsd),
				PX_CUDA_KERNEL_PARAM(prePrepDescd),
				PX_CUDA_KERNEL_PARAM(prepDescd),
				PX_CUDA_KERNEL_PARAM(appliedForced),
				PX_CUDA_KERNEL_PARAM(dt),
				PX_CUDA_KERNEL_PARAM(isTGS),
				PX_CUDA_KERNEL_PARAM(deltaVBuff),
				PX_CUDA_KERNEL_PARAM(sharedDescd)
			};

			const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::PS_COLLISION;
			const PxU32 numBlocks = PxgParticleSystemKernelGridDim::PS_COLLISION;
			CUresult result = mCudaContext->launchKernel(prepPrimitiveCollisionKernelFunction, numBlocks, 1, 1, numThreadsPerBlock, 1, 1, 0, solverStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
			PX_ASSERT(result == CUDA_SUCCESS);
			PX_UNUSED(result);

#if defined(PX_DCU_PORT) && PX_DCU_PORT
			dcuPbdPipelineCheckpoint(mCudaContext, solverStream, "rigid_constraint_prepare_complete",
				totalContactCountsd, 0);
#endif
#if PS_GPU_DEBUG
			result = mCudaContext->streamSynchronize(solverStream);
			if (result != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU ps_contactPrepareLaunch kernel fail!\n");
#endif
		}

		{
			CUdeviceptr contactsd = mPrimitiveContactSortedByParticleBuf.getDevicePtr();
			CUdeviceptr blockOffsetd = mTempBlockCellsHistogramBuf.getDevicePtr();
			CUdeviceptr offsetd = mTempCellsHistogramBuf.getDevicePtr();
			CUdeviceptr pairCountd = mTempHistogramCountBuf.getDevicePtr();
			CUdeviceptr startd = mTempContactBuf.getDevicePtr();
			CUdeviceptr	endd = mTempContactRemapBuf.getDevicePtr();

			#if defined(PX_DCU_PORT) && PX_DCU_PORT
			const bool useFusedParticleRanges = dcuPbdUseFusedParticleRanges();
			const bool useChunkedParticleRanges = useFusedParticleRanges && dcuPbdUseChunkedParticleRanges();
			const bool useParallelParticleRanges = useFusedParticleRanges && dcuPbdUseParallelParticleRanges();
			dcuPbdPrintParticleRangeModeOnce(useFusedParticleRanges, useChunkedParticleRanges, useParallelParticleRanges);
			const PxU32 rangeBlockSize = (useParallelParticleRanges || useChunkedParticleRanges) ? 32u : (useFusedParticleRanges ? 1u : 256u);
			const PxU32 dcuRangeMode = useParallelParticleRanges ? 2u : (useChunkedParticleRanges ? 1u : 0u);
			#else
			const PxU32 rangeBlockSize = PxgParticleSystemKernelBlockDim::ACCUMULATE_DELTA;
			const PxU32 dcuRangeMode = 0;
			#endif

			//compute blockOffset and offset array for particle
			#if defined(PX_DCU_PORT) && PX_DCU_PORT
			if (!useFusedParticleRanges)
			#endif
			{
				const CUfunction findStartEndFirstKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_FIND_RANGESTARTEND_PARTICLE_FIRST);

				PxCudaKernelParam kernelParams[] =
				{
					PX_CUDA_KERNEL_PARAM(particleSystemd),
					PX_CUDA_KERNEL_PARAM(contactsd),
					PX_CUDA_KERNEL_PARAM(totalContactCountsd),
					PX_CUDA_KERNEL_PARAM(blockOffsetd),
					PX_CUDA_KERNEL_PARAM(offsetd),
				};

				const PxU32 numBlocks = PxgParticleSystemKernelGridDim::ACCUMULATE_DELTA;
				CUresult result = mCudaContext->launchKernel(findStartEndFirstKernelFunction, numBlocks, 1, 1, rangeBlockSize, 1, 1, 0, mStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);
				PX_UNUSED(result);
#if PS_GPU_DEBUG
				result = mCudaContext->streamSynchronize(mStream);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU ps_findStartEndParticleFirst kernel fail!\n");

				/*PxU32 totalNumContacts;
				mCudaContext->memcpyDtoH(&totalNumContacts, totalContactCountsd, sizeof(PxU32));

				if (totalNumContacts > 0)
				{
				PxArray<PxU32> offsets;
				offsets.reserve(totalNumContacts);
				offsets.forceSize_Unsafe(totalNumContacts);

				PxU32 blockOffset[32];
				mCudaContext->memcpyDtoH(offsets.begin(), offsetd, sizeof(PxU32) * totalNumContacts);
				mCudaContext->memcpyDtoH(blockOffset, blockOffsetd, sizeof(PxU32) * 32);

				int bob = 0;
				PX_UNUSED(bob);
				}*/
#endif
			}

			//compute start and end range for particle
			{
				const CUfunction findStartEndSecondKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_FIND_RANGESTARTEND_PARTICLE_SECONE);

				PxCudaKernelParam kernelParams[] =
				{
					PX_CUDA_KERNEL_PARAM(contactsd),
					PX_CUDA_KERNEL_PARAM(totalContactCountsd),
					PX_CUDA_KERNEL_PARAM(blockOffsetd),
					PX_CUDA_KERNEL_PARAM(offsetd),
					PX_CUDA_KERNEL_PARAM(pairCountd),
					PX_CUDA_KERNEL_PARAM(startd),
					PX_CUDA_KERNEL_PARAM(endd),
					PX_CUDA_KERNEL_PARAM(dcuRangeMode)
				};

				#if defined(PX_DCU_PORT) && PX_DCU_PORT
				const PxU32 numBlocks = useFusedParticleRanges ? 1u : PxgParticleSystemKernelGridDim::ACCUMULATE_DELTA;
				#else
				const PxU32 numBlocks = PxgParticleSystemKernelGridDim::ACCUMULATE_DELTA;
				#endif
				CUresult result = mCudaContext->launchKernel(findStartEndSecondKernelFunction, numBlocks, 1, 1, rangeBlockSize, 1, 1, 0, mStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);
				PX_UNUSED(result);
#if PS_GPU_DEBUG
				result = mCudaContext->streamSynchronize(mStream);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU ps_findStartEndParticleSecond kernel fail!\n");

				/*PxU32 pairCount;
				mCudaContext->memcpyDtoH(&pairCount, pairCountd, sizeof(PxU32));

				PxArray<PxU32> rangeStart;
				rangeStart.reserve(pairCount);
				rangeStart.forceSize_Unsafe(pairCount);

				PxArray<PxU32> rangeEnd;
				rangeEnd.reserve(pairCount);
				rangeEnd.forceSize_Unsafe(pairCount);

				mCudaContext->memcpyDtoH(rangeStart.begin(), startd, sizeof(PxU32) * pairCount);
				mCudaContext->memcpyDtoH(rangeEnd.begin(), endd, sizeof(PxU32) * pairCount);

				int bob = 0;
				PX_UNUSED(bob);*/

#endif
			}

			#if defined(PX_DCU_PORT) && PX_DCU_PORT
			if (!dcuPbdValidateParticleRanges(mCudaContext, mStream, contactsd, totalContactCountsd,
				pairCountd, startd, endd, mMaxContacts))
				std::abort();
			#endif
		}

#if defined(PX_DCU_PORT) && PX_DCU_PORT
		dcuPbdPipelineCheckpoint(mCudaContext, mStream, "particle_contact_ranges_complete",
			totalContactCountsd, 0);
#endif

	}

	namespace
	{
		PxU32 debugDmaAccumulateRigidDeltasPass1Input(PxArray<PxU64>& rigidIds, PxArray<PxVec4>& deltaVs, 
			const CUdeviceptr numIdsd, const CUdeviceptr rigidIdsd, const CUdeviceptr deltaVd)
		{
			PxU32 numIds;
			cuMemcpyDtoH(&numIds, numIdsd, sizeof(PxU32));
			rigidIds.resize(numIds);
			deltaVs.resize(numIds);
			cuMemcpyDtoH(rigidIds.begin(), rigidIdsd, sizeof(PxU64) * numIds);
			cuMemcpyDtoH(deltaVs.begin(), deltaVd, sizeof(PxVec4) * numIds);
			return numIds;
		}

		void debugComputeAccumulateRigidDeltasPass1(PxArray<PxVec4>& deltaVRef, PxArray<PxVec4>& blockDeltaVRef, PxArray<PxU64>& blockRigidBodyIdRef,
			const PxU32 numBlocks, const PxU32 numThreadsPerBlock, const PxU32 numIds, const PxArray<PxU64>& rigidIds, const PxArray<PxVec4>& deltaVs)
		{
			deltaVRef.resize(numIds);
			blockDeltaVRef.resize(numBlocks, PxVec4(0.0f));
			blockRigidBodyIdRef.resize(numBlocks, 0x8fffffffffffffff);

			//compute reference histogram, with block 
			const PxU32 nbBlocksRequired = (numIds + numThreadsPerBlock - 1) / numThreadsPerBlock;
			const PxU32 nbIterationsPerBlock = (nbBlocksRequired + numBlocks - 1) / numBlocks;

			PxU32 offset = 0;
			PxU32 blockOffset = 0;
			while (offset < numIds)
			{
				PxVec4 deltaPrev = PxVec4(0.0f);
				PxU64 rigidBodyIdPrev = 0x8fffffffffffffff;
				for (PxU32 i = 0; i < nbIterationsPerBlock * numThreadsPerBlock; ++i)
				{
					if (offset < numIds)
					{
						PxVec4 deltaIn = deltaVs[offset];
						PxVec4& deltaOut = deltaVRef[offset];
						PxU64 rigidIdIn = rigidIds[offset];
						if (rigidIdIn != rigidBodyIdPrev)
						{
							deltaPrev = PxVec4(0.0f);
						}
						deltaOut = deltaIn + deltaPrev;
						deltaPrev = deltaOut;
						rigidBodyIdPrev = rigidIdIn;
						offset++;
					}
				}
				if (blockOffset < blockRigidBodyIdRef.size() && offset < numIds)
				{
					blockDeltaVRef[blockOffset] = deltaPrev;
					blockRigidBodyIdRef[blockOffset] = rigidBodyIdPrev;
					blockOffset++;
				}
			}
		}

		PX_FORCE_INLINE bool compare(const PxReal ref, const PxReal cand, const PxReal relEps)
		{
			return PxAbs(1.0f - cand/ref) < relEps;
		}

		PX_FORCE_INLINE bool compare(const PxVec4& ref, const PxVec4& cand, const PxReal relEps)
		{
			return compare(ref.x, cand.x, relEps) &&
				   compare(ref.y, cand.y, relEps) &&
				   compare(ref.z, cand.z, relEps) &&
				   compare(ref.w, cand.w, relEps);
		}

		void debugCompareAccumulateRigidDeltasPass1(const PxArray<PxVec4>& deltaVRef, const PxArray<PxVec4>& blockDeltaVRef, const PxArray<PxU64>& blockRigidIdRef,
			const PxU32 numBlocks, const PxU32 numIds, CUdeviceptr deltaVd, CUdeviceptr blockDeltaVd, CUdeviceptr blockRigidIdd)
		{
			PxArray<PxVec4> deltaVCand(numIds);
			PxArray<PxVec4> blockDeltaVCand(numBlocks);
			PxArray<PxU64> blockRigidIdCand(numBlocks);

			cuMemcpyDtoH(deltaVCand.begin(), deltaVd, sizeof(PxVec4) * numIds);
			cuMemcpyDtoH(blockDeltaVCand.begin(), blockDeltaVd, sizeof(PxVec4) * numBlocks);
			cuMemcpyDtoH(blockRigidIdCand.begin(), blockRigidIdd, sizeof(PxNodeIndex) * numBlocks);

			const PxReal deltaRelEps = 5e-2f; // 5% is very generous, but seems to be needed in some cases
			bool compPassed = true;
			for (PxU32 i = 0; i < numIds; ++i)
			{
				const bool pass = compare(deltaVRef[i], deltaVCand[i], deltaRelEps);
				compPassed &= pass;
			}
			for (PxU32 i = 0; i < numBlocks; ++i)
			{
				const bool pass = compare(blockDeltaVRef[i], blockDeltaVCand[i], deltaRelEps) &&
								  blockRigidIdCand[i] == blockRigidIdRef[i];
				compPassed &= pass;
			}
			if (!compPassed)
			{
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU accumRigidDeltas first kernel fail!\n");
			}
		}

	} // namespace

	void PxgParticleSystemCore::accumulateRigidDeltas(CUdeviceptr prePrepDescd, CUdeviceptr solverCoreDescd, 
		CUdeviceptr sharedDescd, CUdeviceptr artiCoreDescd, CUdeviceptr rigidIdsd, CUdeviceptr numIdsd, CUstream stream,
		const bool isTGS)
	{
#if defined(PX_DCU_PORT) && PX_DCU_PORT
		const PxU32 rigidDeltaBlockSize = 256;
#else
		const PxU32 rigidDeltaBlockSize = PxgParticleSystemKernelBlockDim::ACCUMULATE_DELTA;
#endif
		{
			CUdeviceptr deltaVd = mDeltaVelRigidBuf.getDevicePtr();
			CUdeviceptr blockDeltaVd = mTempBlockDeltaVelBuf.getDevicePtr();
			CUdeviceptr blockRigidIdd = mTempBlockRigidIdBuf.getDevicePtr();

#if PS_GPU_DEBUG
			//copy input data to device for CPU reference code
			PxArray<PxU64> debugRigidIds;
			PxArray<PxVec4> debugDeltaVs;
			PxU32 debugNumIds = debugDmaAccumulateRigidDeltasPass1Input(debugRigidIds, debugDeltaVs,
				numIdsd, rigidIdsd, deltaVd);
#endif

			const CUfunction primitiveCollisionFirstKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::ACCUMULATE_DELTAVEL_RIGIDBODY_FIRST);

			PxCudaKernelParam kernelParams[] =
			{
				PX_CUDA_KERNEL_PARAM(rigidIdsd),
				PX_CUDA_KERNEL_PARAM(numIdsd),
				PX_CUDA_KERNEL_PARAM(deltaVd),
				PX_CUDA_KERNEL_PARAM(blockDeltaVd),
				PX_CUDA_KERNEL_PARAM(blockRigidIdd)
			};

			const PxU32 numBlocks = PxgParticleSystemKernelGridDim::ACCUMULATE_DELTA;

#if PS_GPU_DEBUG
			//compute reference on CPU
			PxArray<PxVec4> debugDeltaVRef;
			PxArray<PxVec4> debugBlockDeltaVRef;
			PxArray<PxU64> debugBlockRigidIdRef;
			debugComputeAccumulateRigidDeltasPass1(debugDeltaVRef, debugBlockDeltaVRef, debugBlockRigidIdRef,
				numBlocks, rigidDeltaBlockSize, debugNumIds, debugRigidIds, debugDeltaVs);

#endif
			CUresult result = mCudaContext->launchKernel(primitiveCollisionFirstKernelFunction, numBlocks, 1, 1, rigidDeltaBlockSize, 1, 1, 0, stream, kernelParams, sizeof(kernelParams), 0, PX_FL);
			PX_ASSERT(result == CUDA_SUCCESS);
			PX_UNUSED(result);

#if PS_GPU_DEBUG
			result = mCudaContext->streamSynchronize(stream);
			if (result != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU accumRigidDeltas first kernel fail!\n");

			debugCompareAccumulateRigidDeltasPass1(debugDeltaVRef, debugBlockDeltaVRef, debugBlockRigidIdRef,
				numBlocks, debugNumIds, deltaVd, blockDeltaVd, blockRigidIdd);
#endif
		}
		{
			CUdeviceptr deltaVd = mDeltaVelRigidBuf.getDevicePtr();
			CUdeviceptr blockDeltaVd = mTempBlockDeltaVelBuf.getDevicePtr();
			CUdeviceptr blockRigidIdd = mTempBlockRigidIdBuf.getDevicePtr();

			const CUfunction primitiveCollisionSecondKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::ACCUMULATE_DELTAVEL_RIGIDBODY_SECOND);

			PxCudaKernelParam kernelParams[] =
			{
				PX_CUDA_KERNEL_PARAM(rigidIdsd),
				PX_CUDA_KERNEL_PARAM(numIdsd),
				PX_CUDA_KERNEL_PARAM(deltaVd),
				PX_CUDA_KERNEL_PARAM(blockDeltaVd),
				PX_CUDA_KERNEL_PARAM(blockRigidIdd),
				PX_CUDA_KERNEL_PARAM(prePrepDescd),
				PX_CUDA_KERNEL_PARAM(solverCoreDescd),
				PX_CUDA_KERNEL_PARAM(artiCoreDescd),
				PX_CUDA_KERNEL_PARAM(sharedDescd),
				PX_CUDA_KERNEL_PARAM(isTGS)
			};

			const PxU32 numBlocks = PxgParticleSystemKernelGridDim::ACCUMULATE_DELTA;
			CUresult result = mCudaContext->launchKernel(primitiveCollisionSecondKernelFunction, numBlocks, 1, 1, rigidDeltaBlockSize, 1, 1, 0, stream, kernelParams, sizeof(kernelParams), 0, PX_FL);
			PX_ASSERT(result == CUDA_SUCCESS);
			PX_UNUSED(result);
#if PS_GPU_DEBUG
			result = mCudaContext->streamSynchronize(stream);
			if (result != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU accumRigidDeltas second kernel fail!\n");
#endif
		}
	}

	void PxgParticleSystemCore::solvePrimitiveCollisionForParticles(CUdeviceptr prePrepDescd, CUdeviceptr solverCoreDescd,
		CUdeviceptr sharedDescd, CUdeviceptr artiCoreDescd, const PxReal dt, bool isTGS, const PxReal coefficient,
		bool isVelIteration)
	{
		//solve particle-primitive collision
		
		CUdeviceptr particleSystemd = getParticleSystemBuffer().getDevicePtr();

		CUdeviceptr totalContactCountsd = getParticleContactCount().getDevicePtr();

		//const PxU32 nbActiveParticleSystems = mSimController->getBodySimManager().mActiveParticleSystems.size();

		//printf("--------------Iter-----------\n");

		//Particle-rigid collisions affecting particles, solved on particle stream!
		{			

			CUdeviceptr contactsd = mPrimitiveContactSortedByParticleBuf.getDevicePtr();
			CUdeviceptr constraintsd = mPrimitiveConstraintSortedByParticleBuf.getDevicePtr();
			CUdeviceptr appliedForced = mPrimitiveConstraintAppliedParticleForces.getDevicePtr();
			CUdeviceptr deltaVd = mDeltaVelParticleBuf.getDevicePtr();
			{
				const CUfunction solvePCParticleKernelFunction = isTGS ? mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_SOLVE_PC_PARTICLE_TGS) : 
					mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_SOLVE_PC_PARTICLE);

				PxCudaKernelParam kernelParams[] =
				{
					PX_CUDA_KERNEL_PARAM(particleSystemd),
					PX_CUDA_KERNEL_PARAM(contactsd),
					PX_CUDA_KERNEL_PARAM(constraintsd),
					PX_CUDA_KERNEL_PARAM(totalContactCountsd),
					PX_CUDA_KERNEL_PARAM(prePrepDescd),
					PX_CUDA_KERNEL_PARAM(solverCoreDescd),
					PX_CUDA_KERNEL_PARAM(sharedDescd),
					PX_CUDA_KERNEL_PARAM(deltaVd),
					PX_CUDA_KERNEL_PARAM(appliedForced),
					PX_CUDA_KERNEL_PARAM(dt),
					PX_CUDA_KERNEL_PARAM(coefficient),
					PX_CUDA_KERNEL_PARAM(isVelIteration),
					PX_CUDA_KERNEL_PARAM(artiCoreDescd)
				};

				const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::PS_COLLISION;
				const PxU32 numBlocks = PxgParticleSystemKernelGridDim::PS_COLLISION;
				CUresult result = mCudaContext->launchKernel(solvePCParticleKernelFunction, numBlocks, 1, 1, numThreadsPerBlock, 1, 1, 0, mStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);
				PX_UNUSED(result);
#if PS_GPU_DEBUG
				result = mCudaContext->streamSynchronize(mStream);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU solvePrimitiveCollisionFirst kernel fail!\n");

				PxU32 totalNumContacts;

				mCudaContext->memcpyDtoH(&totalNumContacts, totalContactCountsd, sizeof(PxU32));

				if (totalNumContacts > 0)
				{
					PxArray<PxVec4> deltaV;
					deltaV.reserve(totalNumContacts);
					deltaV.forceSize_Unsafe(totalNumContacts);

					mCudaContext->memcpyDtoH(deltaV.begin(), deltaVd, sizeof(PxVec4) * totalNumContacts);

					int bob = 0;
					PX_UNUSED(bob);
				}
#endif
				mCudaContext->eventRecord(mSolveParticleRigidEvent, mStream);
			}

			{
				//CUdeviceptr blockOffsetd = mTempBlockCellsHistogramBuf.getDevicePtr();
				//CUdeviceptr offsetd = mTempCellsHistogramBuf.getDevicePtr();
				CUdeviceptr pairCountd = mTempHistogramCountBuf.getDevicePtr();
				CUdeviceptr startd = mTempContactBuf.getDevicePtr();
				CUdeviceptr	endd = mTempContactRemapBuf.getDevicePtr();

				//accumulate deltaV changes for particle
				{
					const CUfunction accumulatedDeltaVKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_ACCUMULATE_DELTAVEL_PARTICLE);

					PxCudaKernelParam kernelParams[] =
					{
						PX_CUDA_KERNEL_PARAM(particleSystemd),
						PX_CUDA_KERNEL_PARAM(contactsd),
						PX_CUDA_KERNEL_PARAM(pairCountd),
						PX_CUDA_KERNEL_PARAM(startd),
						PX_CUDA_KERNEL_PARAM(endd),
						PX_CUDA_KERNEL_PARAM(deltaVd)
					};

					#if defined(PX_DCU_PORT) && PX_DCU_PORT
					const PxU32 numThreadsPerBlock = 256;
					#else
					const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::ACCUMULATE_DELTA;
					#endif
					const PxU32 numBlocks = PxgParticleSystemKernelGridDim::ACCUMULATE_DELTA;
					CUresult result = mCudaContext->launchKernel(accumulatedDeltaVKernelFunction, numBlocks, 1, 1, numThreadsPerBlock, 1, 1, 0, mStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
					PX_ASSERT(result == CUDA_SUCCESS);
					PX_UNUSED(result);
#if PS_GPU_DEBUG
					result = mCudaContext->streamSynchronize(mStream);
					if (result != CUDA_SUCCESS)
						PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU ps_accumulateDeltaVParticleLaunch kernel fail!\n");

					/*PxU32 totalNumContacts;

					mCudaContext->memcpyDtoH(&totalNumContacts, totalContactCountsd, sizeof(PxU32));

					if (totalNumContacts > 0)
					{
					PxArray<PxVec4> deltaV;
					deltaV.reserve(totalNumContacts);
					deltaV.forceSize_Unsafe(totalNumContacts);

					mCudaContext->memcpyDtoH(deltaV.begin(), deltaVd, sizeof(PxVec4) * totalNumContacts);

					int bob = 0;
					PX_UNUSED(bob);
					}*/
#endif
				}

				
			}
		}

		
	}

	//this will be just called by PBD
	void PxgParticleSystemCore::solvePrimitiveCollisionForRigids(CUdeviceptr prePrepDescd, CUdeviceptr solverCoreDescd,
		CUdeviceptr sharedDescd, CUdeviceptr artiCoreDescd, CUstream solverStream, const PxReal dt, bool isTGS, const PxReal coefficient,
		bool isVelIteration)
	{
		//solve particle-primitive collision

		CUdeviceptr particleSystemd = getParticleSystemBuffer().getDevicePtr();

		CUdeviceptr totalContactCountsd = getParticleContactCount().getDevicePtr();

		//Particle-rigid particle collisions, affecting rigids, solved on solverStream
		if (1)
		{
			CUdeviceptr contactsd = mPrimitiveContactSortedByRigidBuf.getDevicePtr();
			CUdeviceptr constraintsd = mPrimitiveConstraintSortedByRigidBuf.getDevicePtr();
			CUdeviceptr appliedForced = mPrimitiveConstraintAppliedRigidForces.getDevicePtr();
			CUdeviceptr deltaVd = mDeltaVelRigidBuf.getDevicePtr();

			const CUfunction solvePCRigidKernelFunction = isTGS ? mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_SOLVE_PC_RIGID_TGS)
				: mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_SOLVE_PC_RIGID);

			PxCudaKernelParam kernelParams[] =
			{
				PX_CUDA_KERNEL_PARAM(particleSystemd),
				PX_CUDA_KERNEL_PARAM(contactsd),
				PX_CUDA_KERNEL_PARAM(constraintsd),
				PX_CUDA_KERNEL_PARAM(totalContactCountsd),
				PX_CUDA_KERNEL_PARAM(prePrepDescd),
				PX_CUDA_KERNEL_PARAM(solverCoreDescd),
				PX_CUDA_KERNEL_PARAM(sharedDescd),
				PX_CUDA_KERNEL_PARAM(deltaVd),
				PX_CUDA_KERNEL_PARAM(appliedForced),
				PX_CUDA_KERNEL_PARAM(dt),
				PX_CUDA_KERNEL_PARAM(coefficient),
				PX_CUDA_KERNEL_PARAM(isVelIteration),
				PX_CUDA_KERNEL_PARAM(artiCoreDescd)
			};

			const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::PS_COLLISION;
			const PxU32 numBlocks = PxgParticleSystemKernelGridDim::PS_COLLISION;
			CUresult result = mCudaContext->launchKernel(solvePCRigidKernelFunction, numBlocks, 1, 1, numThreadsPerBlock, 1, 1, 0, solverStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
			PX_ASSERT(result == CUDA_SUCCESS);
			PX_UNUSED(result);
#if PS_GPU_DEBUG
			result = mCudaContext->streamSynchronize(solverStream);
			if (result != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU ps_solvePCOutputRigidDeltaVLaunch kernel fail!\n");

			/*PxU32 totalNumContacts;

			mCudaContext->memcpyDtoH(&totalNumContacts, totalContactCountsd, sizeof(PxU32));

			if (totalNumContacts > 0)
			{
				PxArray<PxVec4> deltaV;
				deltaV.reserve(totalNumContacts);
				deltaV.forceSize_Unsafe(totalNumContacts);

				mCudaContext->memcpyDtoH(deltaV.begin(), deltaVd, sizeof(PxVec4) * totalNumContacts);

				int bob = 0;
				PX_UNUSED(bob);
			}*/
#endif

			mCudaContext->eventRecord(mSolveRigidParticleEvent, solverStream);

			mCudaContext->streamWaitEvent(solverStream, mSolveParticleRigidEvent);

			
			accumulateRigidDeltas(prePrepDescd, solverCoreDescd, sharedDescd, artiCoreDescd,
				mContactSortedByRigidBuf.getDevicePtr(), totalContactCountsd, solverStream,
				isTGS);
		}

		mCudaContext->streamWaitEvent(mStream, mSolveRigidParticleEvent);

	}


	void PxgParticleSystemCore::solveRigidAttachments(CUdeviceptr prePrepDescd, CUdeviceptr solverCoreDescd, 
		CUdeviceptr sharedDescd, CUdeviceptr artiCoreDescd, CUstream solverStream, const PxReal dt, const bool isTGS,
		const PxReal biasCoefficient, const bool isVelocityIteration,
		CUdeviceptr particleSystemd, CUdeviceptr activeParticleSystemd, const PxU32 nbActiveParticleSystems)
	{
		//const PxU32 nbRigidAttachments = simCore->getNbRigidParticleAttachments();
		const PxU32 nbRigidAttachments = mTotalRigidAttachments;

		if (nbRigidAttachments)
		{

			//CUdeviceptr particleSystemd = simCore->getParticleSystemBuffer().getDevicePtr();

			CUdeviceptr constraintsd = mParticleRigidConstraints.getDevicePtr();
			CUdeviceptr rigidAttachmentIds = mParticleRigidAttachmentIds.getDevicePtr();
			CUdeviceptr deltaVd = mDeltaVelRigidBuf.getDevicePtr();
			CUdeviceptr totalRigidAttachmentsd = mParticleRigidConstraintCount.getDevicePtr();

			{
				const CUfunction solvePCRigidKernelFunction = isTGS ? mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_SOLVE_RIGID_ATTACHMENTS_TGS)
					: mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_SOLVE_RIGID_ATTACHMENTS);

				PxCudaKernelParam kernelParams[] =
				{
					PX_CUDA_KERNEL_PARAM(particleSystemd),
					PX_CUDA_KERNEL_PARAM(constraintsd),
					PX_CUDA_KERNEL_PARAM(nbRigidAttachments),
					PX_CUDA_KERNEL_PARAM(prePrepDescd),
					PX_CUDA_KERNEL_PARAM(solverCoreDescd),
					PX_CUDA_KERNEL_PARAM(sharedDescd),
					PX_CUDA_KERNEL_PARAM(deltaVd),
					PX_CUDA_KERNEL_PARAM(dt),
					PX_CUDA_KERNEL_PARAM(biasCoefficient),
					PX_CUDA_KERNEL_PARAM(isVelocityIteration)
				};

				const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::PS_COLLISION;
				const PxU32 numBlocks = PxgParticleSystemKernelGridDim::PS_COLLISION;
				CUresult result = mCudaContext->launchKernel(solvePCRigidKernelFunction, numBlocks, 1, 1, numThreadsPerBlock, 1, 1, 0, solverStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);
				PX_UNUSED(result);
#if PS_GPU_DEBUG
				result = mCudaContext->streamSynchronize(mStream);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU ps_solvePCOutputRigidDeltaVLaunch kernel fail!\n");

				/*PxU32 totalNumContacts;

				mCudaContext->memcpyDtoH(&totalNumContacts, totalContactCountsd, sizeof(PxU32));

				if (totalNumContacts > 0)
				{
				PxArray<PxVec4> deltaV;
				deltaV.reserve(totalNumContacts);
				deltaV.forceSize_Unsafe(totalNumContacts);

				mCudaContext->memcpyDtoH(deltaV.begin(), deltaVd, sizeof(PxVec4) * totalNumContacts);

				int bob = 0;
				PX_UNUSED(bob);
				}*/
#endif
			}

			/*accumulateRigidDeltas(prePrepDescd, solverCoreDescd, sharedDescd, artiCoreDescd,
				rigidAttachmentIds, totalRigidAttachmentsd, solverStream, true, 1.f, isTGS);*/


			{
				CUdeviceptr blockDeltaVd = mTempBlockDeltaVelBuf.getDevicePtr();
				CUdeviceptr blockRigidIdd = mTempBlockRigidIdBuf.getDevicePtr();

				const CUfunction primitiveCollisionFirstKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::ACCUMULATE_DELTAVEL_RIGIDBODY_FIRST);

				PxCudaKernelParam kernelParams[] =
				{
					PX_CUDA_KERNEL_PARAM(rigidAttachmentIds),
					PX_CUDA_KERNEL_PARAM(totalRigidAttachmentsd),
					PX_CUDA_KERNEL_PARAM(deltaVd),
					PX_CUDA_KERNEL_PARAM(blockDeltaVd),
					PX_CUDA_KERNEL_PARAM(blockRigidIdd)
				};

				const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::ACCUMULATE_DELTA;
				const PxU32 numBlocks = PxgParticleSystemKernelGridDim::ACCUMULATE_DELTA;
				CUresult result = mCudaContext->launchKernel(primitiveCollisionFirstKernelFunction, numBlocks, 1, 1, numThreadsPerBlock, 1, 1, 0, solverStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);
				PX_UNUSED(result);

#if PS_GPU_DEBUG
				result = mCudaContext->streamSynchronize(solverStream);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU accumRigidDeltas first kernel fail!\n");
#endif
			}

			PxReal globalRelaxationCoefficient = 1.f;
			bool useLocalRelax = true;

			{
				CUdeviceptr tempDenomd = mParticleRigidAttachmentScaleBuffer.getDevicePtr();


				const CUfunction primitiveCollisionSecondKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::ACCUMULATE_DELTAVEL_RIGIDBODY_MULTI_CLEAR);

				PxCudaKernelParam kernelParams[] =
				{
					PX_CUDA_KERNEL_PARAM(rigidAttachmentIds),
					PX_CUDA_KERNEL_PARAM(totalRigidAttachmentsd),
					PX_CUDA_KERNEL_PARAM(prePrepDescd),
					PX_CUDA_KERNEL_PARAM(solverCoreDescd),
					PX_CUDA_KERNEL_PARAM(artiCoreDescd),
					PX_CUDA_KERNEL_PARAM(tempDenomd)
				};

				const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::ACCUMULATE_DELTA;
				const PxU32 numBlocks = PxgParticleSystemKernelGridDim::ACCUMULATE_DELTA;
				CUresult result = mCudaContext->launchKernel(primitiveCollisionSecondKernelFunction, numBlocks, 1, 1, numThreadsPerBlock, 1, 1, 0, solverStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);
				PX_UNUSED(result);
#if PS_GPU_DEBUG
				result = mCudaContext->streamSynchronize(solverStream);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU accumRigidDeltas second kernel fail!\n");
#endif
			}

			{
				CUdeviceptr blockDeltaVd = mTempBlockDeltaVelBuf.getDevicePtr();
				CUdeviceptr blockRigidIdd = mTempBlockRigidIdBuf.getDevicePtr();
				CUdeviceptr tempDenomd = mParticleRigidAttachmentScaleBuffer.getDevicePtr();



				const CUfunction primitiveCollisionSecondKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::ACCUMULATE_DELTAVEL_RIGIDBODY_SECOND_MULTI1);

				PxCudaKernelParam kernelParams[] =
				{
					PX_CUDA_KERNEL_PARAM(rigidAttachmentIds),
					PX_CUDA_KERNEL_PARAM(totalRigidAttachmentsd),
					PX_CUDA_KERNEL_PARAM(deltaVd),
					PX_CUDA_KERNEL_PARAM(blockDeltaVd),
					PX_CUDA_KERNEL_PARAM(blockRigidIdd),
					PX_CUDA_KERNEL_PARAM(prePrepDescd),
					PX_CUDA_KERNEL_PARAM(solverCoreDescd),
					PX_CUDA_KERNEL_PARAM(artiCoreDescd),
					PX_CUDA_KERNEL_PARAM(sharedDescd),
					PX_CUDA_KERNEL_PARAM(tempDenomd),
					PX_CUDA_KERNEL_PARAM(useLocalRelax),
					PX_CUDA_KERNEL_PARAM(globalRelaxationCoefficient),
					PX_CUDA_KERNEL_PARAM(isTGS)
				};

				const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::ACCUMULATE_DELTA;
				const PxU32 numBlocks = PxgParticleSystemKernelGridDim::ACCUMULATE_DELTA;
				CUresult result = mCudaContext->launchKernel(primitiveCollisionSecondKernelFunction, numBlocks, 1, 1, numThreadsPerBlock, 1, 1, 0, solverStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);
				PX_UNUSED(result);
#if PS_GPU_DEBUG
				result = mCudaContext->streamSynchronize(solverStream);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU accumRigidDeltas second kernel fail!\n");
#endif
			}

			{
				CUdeviceptr blockDeltaVd = mTempBlockDeltaVelBuf.getDevicePtr();
				CUdeviceptr blockRigidIdd = mTempBlockRigidIdBuf.getDevicePtr();
				CUdeviceptr tempDenomd = mParticleRigidAttachmentScaleBuffer.getDevicePtr();



				const CUfunction primitiveCollisionSecondKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::ACCUMULATE_DELTAVEL_RIGIDBODY_SECOND_MULTI2);

				PxCudaKernelParam kernelParams[] =
				{
					PX_CUDA_KERNEL_PARAM(rigidAttachmentIds),
					PX_CUDA_KERNEL_PARAM(totalRigidAttachmentsd),
					PX_CUDA_KERNEL_PARAM(deltaVd),
					PX_CUDA_KERNEL_PARAM(blockDeltaVd),
					PX_CUDA_KERNEL_PARAM(blockRigidIdd),
					PX_CUDA_KERNEL_PARAM(prePrepDescd),
					PX_CUDA_KERNEL_PARAM(solverCoreDescd),
					PX_CUDA_KERNEL_PARAM(artiCoreDescd),
					PX_CUDA_KERNEL_PARAM(sharedDescd),
					PX_CUDA_KERNEL_PARAM(tempDenomd),
					PX_CUDA_KERNEL_PARAM(useLocalRelax),
					PX_CUDA_KERNEL_PARAM(globalRelaxationCoefficient),
					PX_CUDA_KERNEL_PARAM(isTGS)
				};

				const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::ACCUMULATE_DELTA;
				const PxU32 numBlocks = PxgParticleSystemKernelGridDim::ACCUMULATE_DELTA;
				CUresult result = mCudaContext->launchKernel(primitiveCollisionSecondKernelFunction, numBlocks, 1, 1, numThreadsPerBlock, 1, 1, 0, solverStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);
				PX_UNUSED(result);
#if PS_GPU_DEBUG
				result = mCudaContext->streamSynchronize(solverStream);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU accumRigidDeltas second kernel fail!\n");
#endif
			}


			/*const PxU32 nbActiveParticleSystems = mSimController->getBodySimManager().mActiveParticleSystems.size();
			PxgSimulationCore* core = mSimController->getSimulationCore();

			CUdeviceptr activeParticleSystemd = core->getActiveParticleSystemBuffer().getDevicePtr();*/

			applyDeltas(particleSystemd, activeParticleSystemd, nbActiveParticleSystems, dt, solverStream);

		}
	}



	void PxgParticleSystemCore::prepRigidAttachments(CUdeviceptr prePrepDescd, CUdeviceptr prepDescd,
		bool isTGS, const PxReal dt, CUstream stream, const PxU32 nbActiveParticleSystems, CUdeviceptr activeParticleSystemsD,
		PxU32 numSolverBodies)
	{
		if (mMaxRigidAttachmentsPerSystem)
		{
			mParticleRigidAttachmentScaleBuffer.allocate(sizeof(PxReal) * numSolverBodies, PX_FL);
			CUdeviceptr particleSystemd = getParticleSystemBuffer().getDevicePtr();

			CUdeviceptr constraintsd = mParticleRigidConstraints.getDevicePtr(); 
			CUdeviceptr rigidAttachmentIds = mParticleRigidAttachmentIds.getDevicePtr();

			PxU32 maxRigidAttachments = mMaxRigidAttachmentsPerSystem;
			PxU32 nbSystems = nbActiveParticleSystems;

			//prepare primitive constraints sorted by particle id
			{
				const CUfunction prepAttachmentKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_PREP_RIGID_ATTACHMENTS);

				PxCudaKernelParam kernelParams[] =
				{
					PX_CUDA_KERNEL_PARAM(particleSystemd),
					PX_CUDA_KERNEL_PARAM(rigidAttachmentIds),
					PX_CUDA_KERNEL_PARAM(constraintsd),
					PX_CUDA_KERNEL_PARAM(prePrepDescd),
					PX_CUDA_KERNEL_PARAM(prepDescd),
					PX_CUDA_KERNEL_PARAM(dt),
					PX_CUDA_KERNEL_PARAM(isTGS),
					PX_CUDA_KERNEL_PARAM(activeParticleSystemsD)
				};

				const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::PS_COLLISION;
				//const PxU32 numBlocks = PxgParticleSystemKernelGridDim::PS_COLLISION;
				const PxU32 numBlocks = (maxRigidAttachments + PxgParticleSystemKernelBlockDim::PS_COLLISION - 1) / PxgParticleSystemKernelBlockDim::PS_COLLISION;
				CUresult result = mCudaContext->launchKernel(prepAttachmentKernelFunction, numBlocks, nbSystems, 1, numThreadsPerBlock, 1, 1, 0, stream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);
				PX_UNUSED(result);
#if PS_GPU_DEBUG
				result = mCudaContext->streamSynchronize(mStream);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU prepRigidAttachments ps_rigidAttachmentPrepareLaunch kernel fail!\n");
				PX_ASSERT(result == CUDA_SUCCESS);
#endif
			}
		}
	}

	void PxgParticleSystemCore::prepParticleConstraint(CUdeviceptr prePrepDescd, CUdeviceptr prepDescd, CUdeviceptr sharedDescd,
		bool isTGS, const PxReal dt)
	{
		CUdeviceptr particleSystemd = getParticleSystemBuffer().getDevicePtr();

		CUdeviceptr totalContactCountsd = getParticleContactCount().getDevicePtr();
		CUdeviceptr contactsd = mPrimitiveContactSortedByParticleBuf.getDevicePtr();
		CUdeviceptr constraintsd = mPrimitiveConstraintSortedByParticleBuf.getDevicePtr();
		CUdeviceptr appliedForced = mPrimitiveConstraintAppliedParticleForces.getDevicePtr();
		void* nullContactDeltaV = NULL;

		//prepare primitive constraints sorted by particle id
		{
			const CUfunction prepPrimitiveCollisionKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_CONTACT_PREPARE);

			PxCudaKernelParam kernelParams[] =
			{
				PX_CUDA_KERNEL_PARAM(particleSystemd),
				PX_CUDA_KERNEL_PARAM(contactsd),
				PX_CUDA_KERNEL_PARAM(totalContactCountsd),
				PX_CUDA_KERNEL_PARAM(constraintsd),
				PX_CUDA_KERNEL_PARAM(prePrepDescd),
				PX_CUDA_KERNEL_PARAM(prepDescd),
				PX_CUDA_KERNEL_PARAM(appliedForced),
				PX_CUDA_KERNEL_PARAM(dt),
				PX_CUDA_KERNEL_PARAM(isTGS),
				PX_CUDA_KERNEL_PARAM(nullContactDeltaV),
				PX_CUDA_KERNEL_PARAM(sharedDescd)
			};

			const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::PS_COLLISION;
			const PxU32 numBlocks = PxgParticleSystemKernelGridDim::PS_COLLISION;
			CUresult result = mCudaContext->launchKernel(prepPrimitiveCollisionKernelFunction, numBlocks, 1, 1, numThreadsPerBlock, 1, 1, 0, mStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
			PX_ASSERT(result == CUDA_SUCCESS);
			PX_UNUSED(result);
#if PS_GPU_DEBUG
			result = mCudaContext->streamSynchronize(mStream);
			if (result != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU prepParticleConstraint ps_contactPrepareLaunch kernel fail!\n");
			PX_ASSERT(result == CUDA_SUCCESS);
#endif
		}
	}
	
	void PxgParticleSystemCore::stepParticleSystems(CUdeviceptr particleSystemsd, CUdeviceptr activeParticleSystemsd, const PxU32 nbActiveParticleSystems,
		const PxReal dt, const PxReal totalInvDt, bool isFirstIteration)
	{
		const bool isTGS = mGpuContext->isTGS();
		const bool externalForcesEveryTgsIterationEnabled = mGpuContext->isExternalForcesEveryTgsIterationEnabled();

		if(mMaxParticles > 0)
		{
			PxCudaKernelParam kernelParams[] =
			{
				PX_CUDA_KERNEL_PARAM(particleSystemsd),
				PX_CUDA_KERNEL_PARAM(activeParticleSystemsd),
				PX_CUDA_KERNEL_PARAM(dt),
				PX_CUDA_KERNEL_PARAM(totalInvDt),
				PX_CUDA_KERNEL_PARAM(isFirstIteration),
				PX_CUDA_KERNEL_PARAM(isTGS),
			    PX_CUDA_KERNEL_PARAM(externalForcesEveryTgsIterationEnabled),
				PX_CUDA_KERNEL_PARAM(mGravity)
			};

			CUfunction stepKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_STEP_PARTICLES);
			{
				const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::PS_SOLVE;
				const PxU32 numBlocks = (mMaxParticles + numThreadsPerBlock - 1) / numThreadsPerBlock;
				CUresult result = mCudaContext->launchKernel(stepKernelFunction, numBlocks, nbActiveParticleSystems, 1, numThreadsPerBlock, 1, 1, 0, mStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);
				PX_UNUSED(result);
			}

#if PS_GPU_DEBUG
			CUresult result = mCudaContext->streamSynchronize(mStream);
			PX_ASSERT(result == CUDA_SUCCESS);
			if (result != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU ps_stepParticleSystemLaunch kernel fail!\n");
#endif
		}

	}

	void PxgParticleSystemCore::integrateSystem(CUdeviceptr particleSystemsd, CUdeviceptr activeParticleSystemd, const PxU32 nbActiveParticleSystems, const PxReal dt, const PxReal epsilonSq)
	{
		const PxU32 maxParticles = getMaxParticles();

		if (maxParticles)
		{
			const CUfunction integrateKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_INTEGRATION);
			
			{
				const PxU32 numBlocks = (maxParticles + PxgParticleSystemKernelBlockDim::UPDATEBOUND - 1) / PxgParticleSystemKernelBlockDim::UPDATEBOUND;

				PxCudaKernelParam kernelParams[] =
				{
					PX_CUDA_KERNEL_PARAM(particleSystemsd),
					PX_CUDA_KERNEL_PARAM(activeParticleSystemd),
					PX_CUDA_KERNEL_PARAM(dt),
					PX_CUDA_KERNEL_PARAM(epsilonSq)
				};

				CUresult result = mCudaContext->launchKernel(integrateKernelFunction, numBlocks, nbActiveParticleSystems, 1, PxgParticleSystemKernelBlockDim::UPDATEBOUND, 1, 1, 0, mFinalizeStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);

				PX_UNUSED(result);

#if PS_GPU_DEBUG
				result = mCudaContext->streamSynchronize(mFinalizeStream);
				PX_ASSERT(result == CUDA_SUCCESS);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU integrateSystem kernel fail!\n");
#endif
			}
		}
	}

	void PxgParticleSystemCore::copyUnsortedArrayToUserBuffer(CUdeviceptr particleSystemsd, CUdeviceptr activeParticleSystemsd, const PxU32 nbActiveParticles)
	{
		static PxU32 count = 0;
		count++;

		const CUfunction preIntegrateKernelFunction = mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_UPDATE_BUFFER_DATA);

#if defined(PX_DCU_PORT) && PX_DCU_PORT
		const PxU32 numThreadsPerBlock = 256;
#else
		const PxU32 numThreadsPerBlock = PxgParticleSystemKernelBlockDim::UPDATEBOUND;
#endif
		const PxU32 numBlocks = (mMaxParticlesPerBuffer + numThreadsPerBlock - 1) / numThreadsPerBlock;

		if (numBlocks)
		{
			PxCudaKernelParam kernelParams[] =
			{
				PX_CUDA_KERNEL_PARAM(particleSystemsd),
				PX_CUDA_KERNEL_PARAM(activeParticleSystemsd),
				PX_CUDA_KERNEL_PARAM(count)
			};


			CUresult result = mCudaContext->launchKernel(preIntegrateKernelFunction, numBlocks, mMaxBuffersPerSystem, nbActiveParticles, numThreadsPerBlock, 1, 1, 0, mFinalizeStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
			PX_ASSERT(result == CUDA_SUCCESS);
			PX_UNUSED(result);


#if PS_GPU_DEBUG
			result = mCudaContext->streamSynchronize(mFinalizeStream);
			PX_ASSERT(result == CUDA_SUCCESS);
			if (result != CUDA_SUCCESS)
				PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU ps_updateBufferLaunch kernel fail!\n");
#endif
		}
	}

	void PxgParticleSystemCore::copyUserBufferDataToHost(PxgParticleSystem* particleSystems, PxU32* activeParticleSystems, PxU32 nbActiveParticleSystems)
	{
		const PxU32* particleSystemNodeIndex = mParticleSystemNodeIndexPool.begin();
		void** bodySimsLL = mSimController->getBodySimManager().mBodies.begin();

		for (PxU32 i = 0; i < nbActiveParticleSystems; ++i)
		{
			const PxU32 index = activeParticleSystems[i];
			PxgParticleSystem& particleSystem = particleSystems[index];
			Dy::ParticleSystem* dyParticleSystem = reinterpret_cast<Dy::ParticleSystem*>(bodySimsLL[particleSystemNodeIndex[index]]);
			Dy::ParticleSystemCore& dyParticleSystemCore = dyParticleSystem->getCore();

			if ((dyParticleSystem->mFlag & Dy::ParticleSystemFlag::eENABLE_GPU_DATA_SYNC) && (particleSystem.mCommonData.mNumParticles > 0))
			{
				for (PxU32 b = 0; b < dyParticleSystemCore.mParticleBuffers.size(); ++b)
				{
					PxgParticleBuffer* particleBuffer = static_cast<PxgParticleBuffer*>(dyParticleSystemCore.mParticleBuffers[b]);
					particleBuffer->copyToHost(mFinalizeStream);
				}
				for (PxU32 b = 0; b < dyParticleSystemCore.mParticleDiffuseBuffers.size(); ++b)
				{
					PxgParticleAndDiffuseBuffer* particleDiffuseBuffer = static_cast<PxgParticleAndDiffuseBuffer*>(dyParticleSystemCore.mParticleDiffuseBuffers[b]);
					particleDiffuseBuffer->copyToHost(mFinalizeStream);
				}
				for (PxU32 b = 0; b < dyParticleSystemCore.mParticleClothBuffers.size(); ++b)
				{
					PxgParticleClothBuffer* particleClothBuffer = static_cast<PxgParticleClothBuffer*>(dyParticleSystemCore.mParticleClothBuffers[b]);
					particleClothBuffer->copyToHost(mFinalizeStream);
				}
				for (PxU32 b = 0; b < dyParticleSystemCore.mParticleRigidBuffers.size(); ++b)
				{
					PxgParticleRigidBuffer* particleRigidBuffer = static_cast<PxgParticleRigidBuffer*>(dyParticleSystemCore.mParticleRigidBuffers[b]);
					particleRigidBuffer->copyToHost(mFinalizeStream);
				}
			}
		}
	}

	void PxgParticleSystemCore::getMaxIterationCount(PxgBodySimManager& bodySimManager, const PxU32 nbActiveParticles, const PxU32* activeParticles, PxI32& maxPosIters, PxI32& maxVelIters)
	{
		void** bodySimsLL = bodySimManager.mBodies.begin();

		PxU32* particleNodeIndex = mParticleSystemNodeIndexPool.begin();

		for (PxU32 i = 0; i < nbActiveParticles; ++i)
		{
			const PxU32 index = activeParticles[i];
			const PxU32 nodeIdex = particleNodeIndex[index];
			Dy::ParticleSystem* dyParticleSystem = reinterpret_cast<Dy::ParticleSystem*>(bodySimsLL[nodeIdex]);

			PxU16 solverIterationCounts = dyParticleSystem->getIterationCounts();

			maxPosIters = PxMax(PxI32(solverIterationCounts & 0xff), maxPosIters);
			maxVelIters = PxMax(PxI32(solverIterationCounts >> 8), maxVelIters);
		}
	}

	void PxgParticleSystemCore::updateParticleSystemData(PxgParticleSystem& sys, Dy::ParticleSystemCore& dyParticleSystemCore)
	{
		const PxReal particleContactDistance = 2.f * dyParticleSystemCore.particleContactOffset;
		const PxReal fluidRestDistance = 2.f * dyParticleSystemCore.fluidRestOffset;

		if (dyParticleSystemCore.particleContactOffset_prev != dyParticleSystemCore.particleContactOffset ||
			dyParticleSystemCore.fluidRestOffset_prev != dyParticleSystemCore.fluidRestOffset)
		{
			PxReal fluidRestDensity;
			PxReal fluidDensityConstraintScale;
			PxReal fluidSurfaceConstraintScale;

			// TOFIX: Hack params to known values
			CalculateRestDensity(
				fluidRestDistance,
				particleContactDistance,
				fluidRestDensity,
				fluidDensityConstraintScale,
				fluidSurfaceConstraintScale
			);

			sys.mCommonData.mParticleContactDistance = particleContactDistance;
			sys.mCommonData.mParticleContactDistanceInv = 1.0f / particleContactDistance;
			sys.mCommonData.mParticleContactDistanceSq = particleContactDistance * particleContactDistance;

			// norm for visocelastic spike (from viscoelastic fluids)
			sys.mData.mSpiky1 = 15.0f / (kPi * particleContactDistance * particleContactDistance * particleContactDistance);
			sys.mData.mSpiky2 = 30.0f / (kPi * particleContactDistance * particleContactDistance * particleContactDistance * particleContactDistance);

			sys.mData.mRestDensity = fluidRestDensity;
			sys.mData.mFluidSurfaceConstraintScale = fluidSurfaceConstraintScale;
			sys.mData.mInvRestDensity = 1.0f / fluidRestDensity;
			sys.mData.mLambdaScale = 1.0f / fluidDensityConstraintScale;

			dyParticleSystemCore.particleContactOffset_prev = dyParticleSystemCore.particleContactOffset;
			dyParticleSystemCore.fluidRestOffset_prev = dyParticleSystemCore.fluidRestOffset;
		}
		sys.mData.mRestDensityBoundary = sys.mData.mRestDensity*dyParticleSystemCore.fluidBoundaryDensityScale;

		sys.mData.mRestOffset = dyParticleSystemCore.restOffset;
		sys.mData.mFluidRestOffset = dyParticleSystemCore.fluidRestOffset;
		sys.mData.mSolidRestOffset = dyParticleSystemCore.solidRestOffset;

		//data.mRelaxationFactor = 1.0f / (1.0f + relaxationFactor);
		sys.mData.mRelaxationFactor = 1.0f;

		sys.mData.mFlags = dyParticleSystemCore.mFlags;
	    sys.mData.mLockFlags = PxU16(dyParticleSystemCore.mLockFlags);

		sys.mData.mWind = dyParticleSystemCore.mWind;

		sys.mData.mNumPhaseToMaterials = dyParticleSystemCore.mPhaseGroupToMaterialHandle.size();

		sys.mData.mMaxVelocity = dyParticleSystemCore.maxVelocity;
	}

	void PxgParticleSystemCore::gpuDMAActiveParticleIndices(const PxU32* activeParticleSystems, const PxU32 numActiveParticleSystems, CUstream stream)
	{
		PX_PROFILE_ZONE("gpuDMAActiveParticleIndices", 0);
		mActiveParticleSystemBuffer.allocate(sizeof(PxU32) * numActiveParticleSystems, PX_FL);
		mCudaContext->memcpyHtoDAsync(mActiveParticleSystemBuffer.getDevicePtr(), activeParticleSystems, sizeof(PxU32) * numActiveParticleSystems, stream);
		
	}

	void PxgParticleSystemCore::releaseInternalParticleSystemDataBuffer()
	{
		for (PxU32 i = 0; i < mParticleSystemDataBuffer.size(); ++i)
		{
			if (mParticleSystemDataBuffer[i])
			{
				mParticleSystemDataBuffer[i]->~PxgParticleSystemBuffer();
				PX_FREE(mParticleSystemDataBuffer[i]);
			}
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////

	PxgDiffuseParticleCore::PxgDiffuseParticleCore(PxgEssentialCore* core) :
		mEssentialCore(core),
		mDiffuseParticlesRandomTableBuf(core->mHeapMemoryManager, PxsHeapStats::eSHARED_PARTICLES)
	{
		mRandomTableSize = 1024;
		mMaxDiffuseParticles = 0;
	}

	PxgDiffuseParticleCore::~PxgDiffuseParticleCore()
	{
	}

	void PxgDiffuseParticleCore::releaseInternalDiffuseParticleDataBuffer()
	{
		for (PxU32 i = 0; i < mDiffuseParticleDataBuffer.size(); ++i)
		{
			if (mDiffuseParticleDataBuffer[i])
			{
				mDiffuseParticleDataBuffer[i]->~PxgParticleSystemDiffuseBuffer();
				PX_FREE(mDiffuseParticleDataBuffer[i]);
			}
		}
	}

	void PxgDiffuseParticleCore::preDiffuseIntegrateSystem(CUdeviceptr particleSystemsd, CUdeviceptr activeParticleSystemsd, const PxU32 nbActiveParticles, const PxVec3 gravity,
		const PxReal dt, CUstream bpStream)
	{
		if (mMaxDiffuseParticles)
		{
			
			const CUfunction preIntegrateKernelFunction = mEssentialCore->mGpuKernelWranglerManager->getKernelWrangler()->getCuFunction(PxgKernelIds::PS_PRE_DIFFUSE_INTEGRATION);

			const PxU32 numBlocks = (mMaxDiffuseParticles + PxgParticleSystemKernelBlockDim::UPDATEBOUND - 1) / PxgParticleSystemKernelBlockDim::UPDATEBOUND;

			{
				PxCudaKernelParam kernelParams[] =
				{
					PX_CUDA_KERNEL_PARAM(particleSystemsd),
					PX_CUDA_KERNEL_PARAM(activeParticleSystemsd),
					PX_CUDA_KERNEL_PARAM(gravity),
					PX_CUDA_KERNEL_PARAM(dt)
				};

				//one  blockIdx.z to indicate normal particles or diffuse particles
				CUresult result = mEssentialCore->mCudaContext->launchKernel(preIntegrateKernelFunction, numBlocks, nbActiveParticles, 1, PxgParticleSystemKernelBlockDim::UPDATEBOUND, 1, 1, 0, bpStream, kernelParams, sizeof(kernelParams), 0, PX_FL);
				PX_ASSERT(result == CUDA_SUCCESS);
				PX_UNUSED(result);


#if PS_GPU_DEBUG
				result = mEssentialCore->mCudaContext->streamSynchronize(bpStream);
				PX_ASSERT(result == CUDA_SUCCESS);
				if (result != CUDA_SUCCESS)
					PxGetFoundation().error(PxErrorCode::eINTERNAL_ERROR, PX_FL, "GPU ps_preDiffuseIntegrateLaunch kernel fail!\n");
#endif
			}
		}

	}

	void PxgDiffuseParticleCore::resizeDiffuseParticleParticleBuffers(PxgParticleSystem& particleSystem, PxgParticleSystemDiffuseBuffer* buffer, const PxU32 numParticles)
	{
		buffer->diffuse_potentials.allocate(numParticles * sizeof(float2), PX_FL);

		particleSystem.mDiffusePotentials = reinterpret_cast<float2*>(buffer->diffuse_potentials.getDevicePtr());
	}

}

//#pragma optimize("", on)
