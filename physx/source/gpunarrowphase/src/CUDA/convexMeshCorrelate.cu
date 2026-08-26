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

#include "foundation/PxMath.h"
#include "foundation/PxVec3.h"
#include "foundation/PxTransform.h"

#include "AlignedTransform.h"

#include "PxgContactManager.h"
#include "PxgNpKernelIndices.h"
#include "PxgPersistentContactManifold.h"
#include "PxsContactManagerState.h"

#include "PxgCommonDefines.h"
#include "reduction.cuh"
#include "contactReduction.cuh"

using namespace physx;

extern "C" __host__ void initNarrowphaseKernels3() {}

#if defined(PX_DCU_PORT) && PX_DCU_PORT
extern "C" __device__ __constant__ char gPxgDcuCorrelateStageVersion[] = "PX_DCU_CORE_STAGE_V25_FINITE_SERIAL_CORRELATE";
#endif

#include "manifold.cuh"
#include "nputils.cuh"


#include "cudaNpCommon.h"
#include "convexNpCommon.h"


//This method compacts the depthBuffer + nicBuffer to only include the set of triangles that actually have contacts.
//These then must be sorted based on depth values to construct patches, then appended and then output
__device__ PX_FORCE_INLINE
int purgeEmptyTriangles(const PxU32 syncMask, PxReal* depthBuffer, ConvexTriNormalAndIndex* nicBuffer, int count, PxU32* tempCounts)
{
	const int tI = threadIdx.x;

	PxU32 nbTriangles = 0;

	int loaded = 0;
	for (; loaded < count; loaded += 32)
	{
		int nbToLoad = PxMin(count - loaded, 32);

		const bool inRange = tI < nbToLoad;
		PxReal d = inRange ? depthBuffer[loaded + tI] : FLT_MAX;
		PxU32 nbContacts = inRange ? ConvexTriNormalAndIndex::getNbContacts(nicBuffer[loaded + tI].index) : 0;
		int valid = d != FLT_MAX && nbContacts > 0;
		int validBits = __ballot_sync(syncMask, valid);
		int dest = __popc(validBits & ((PxU32(1) << tI) - 1)) + nbTriangles;

		if (valid)
		{
			depthBuffer[dest] = d;
			tempCounts[dest] = ((loaded + tI) << 4) | nbContacts;
			nicBuffer[dest] = nicBuffer[loaded + tI];
		}

		nbTriangles += __popc(validBits);
	}

	return nbTriangles;
}

struct Scratch
{
	volatile ConvexMeshPair pair;
	int pairIndex;
};

#if defined(PX_DCU_PORT) && PX_DCU_PORT
struct DcuCorrelateContact
{
	float4 pointSeparation;
	PxU32 triangleIndex;
};

__device__ PX_FORCE_INLINE bool isValidDcuCorrelateNormal(const PxVec3& normal)
{
	const PxReal magnitudeSquared = normal.magnitudeSquared();
	return normal.isFinite() && PxIsFinite(magnitudeSquared) && magnitudeSquared >= PX_NORMALIZATION_EPSILON;
}

__device__ PX_FORCE_INLINE bool isValidDcuCorrelateContact(const float4& pointSeparation)
{
	return PxIsFinite(pointSeparation.x) && PxIsFinite(pointSeparation.y) &&
		PxIsFinite(pointSeparation.z) && PxIsFinite(pointSeparation.w);
}

__device__ PX_FORCE_INLINE PxVec3 projectDcuCorrelatePoint(const DcuCorrelateContact& contact, const PxVec3& normal)
{
	const PxVec3 point(contact.pointSeparation.x, contact.pointSeparation.y, contact.pointSeparation.z);
	return point - normal * point.dot(normal);
}

__device__ PX_FORCE_INLINE PxU32 selectDcuCorrelateMax(
	const DcuCorrelateContact* contacts, const PxU32 mask, const PxVec3& normal,
	const PxU32 referenceIndex, const PxU32 metric)
{
	PxU32 selected = WARP_SIZE;
	PxReal selectedValue = -PX_MAX_F32;
	const PxVec3 referencePoint = referenceIndex < WARP_SIZE
		? projectDcuCorrelatePoint(contacts[referenceIndex], normal) : PxVec3(0.f);

	for (PxU32 scanMask = mask; scanMask; scanMask = clearLowestSetBit(scanMask))
	{
		const PxU32 index = lowestSetIndex(scanMask);
		const PxReal value = metric == 0 ? contacts[index].pointSeparation.w
			: (projectDcuCorrelatePoint(contacts[index], normal) - referencePoint).magnitude();

		if (selected == WARP_SIZE || value > selectedValue || (value == selectedValue && index < selected))
		{
			selected = index;
			selectedValue = value;
		}
	}
	return selected;
}

__device__ PX_FORCE_INLINE PxU32 selectDcuCorrelateMinSeparation(
	const DcuCorrelateContact* contacts, const PxU32 mask, const PxU32 anchorMask, const PxReal clusterBias)
{
	PxU32 selected = WARP_SIZE;
	PxReal selectedValue = PX_MAX_F32;
	for (PxU32 scanMask = mask; scanMask; scanMask = clearLowestSetBit(scanMask))
	{
		const PxU32 index = lowestSetIndex(scanMask);
		const PxReal value = contacts[index].pointSeparation.w - ((anchorMask & (PxU32(1) << index)) ? clusterBias : 0.f);
		if (selected == WARP_SIZE || value < selectedValue || (value == selectedValue && index < selected))
		{
			selected = index;
			selectedValue = value;
		}
	}
	return selected;
}

__device__ PX_FORCE_INLINE PxU32 reduceDcuCorrelateContacts(
	DcuCorrelateContact* contacts, const PxU32 allMask, const PxVec3& normal, const PxReal clusterBias)
{
	if (__popc(allMask) <= SUBMANIFOLD_MAX_CONTACTS)
		return allMask;

	const PxU32 i0 = selectDcuCorrelateMax(contacts, allMask, normal, WARP_SIZE, 0);
	PxU32 anchorMask = PxU32(1) << i0;
	const PxU32 i1 = selectDcuCorrelateMax(contacts, allMask & ~anchorMask, normal, i0, 1);
	anchorMask |= PxU32(1) << i1;

	const PxVec3 point0 = projectDcuCorrelatePoint(contacts[i0], normal);
	const PxVec3 direction = normal.cross(projectDcuCorrelatePoint(contacts[i1], normal) - point0);
	PxU32 maxIndex = WARP_SIZE;
	PxU32 minIndex = WARP_SIZE;
	PxReal maxValue = -PX_MAX_F32;
	PxReal minValue = PX_MAX_F32;
	for (PxU32 scanMask = allMask & ~anchorMask; scanMask; scanMask = clearLowestSetBit(scanMask))
	{
		const PxU32 index = lowestSetIndex(scanMask);
		const PxReal value = direction.dot(projectDcuCorrelatePoint(contacts[index], normal) - point0);
		if (maxIndex == WARP_SIZE || value > maxValue || (value == maxValue && index < maxIndex))
		{
			maxIndex = index;
			maxValue = value;
		}
	}
	anchorMask |= PxU32(1) << maxIndex;
	for (PxU32 scanMask = allMask & ~anchorMask; scanMask; scanMask = clearLowestSetBit(scanMask))
	{
		const PxU32 index = lowestSetIndex(scanMask);
		const PxReal value = direction.dot(projectDcuCorrelatePoint(contacts[index], normal) - point0);
		if (minIndex == WARP_SIZE || value < minValue || (value == minValue && index < minIndex))
		{
			minIndex = index;
			minValue = value;
		}
	}
	anchorMask |= PxU32(1) << minIndex;

	PxU32 selectedMask = 0;
	PxU32 remainingAnchors = anchorMask;
	PxU32 clusterIndex = 0;
	while (remainingAnchors)
	{
		PxU32 clusterMask = 0;
		for (PxU32 scanMask = allMask; scanMask; scanMask = clearLowestSetBit(scanMask))
		{
			const PxU32 index = lowestSetIndex(scanMask);
			const PxVec3 point = projectDcuCorrelatePoint(contacts[index], normal);
			PxReal closestDistance = PX_MAX_F32;
			PxU32 closestCluster = 0;
			PxU32 candidateAnchors = anchorMask;
			PxU32 candidateCluster = 0;
			while (candidateAnchors)
			{
				const PxU32 candidateAnchor = lowestSetIndex(candidateAnchors);
				const PxReal distance = (point - projectDcuCorrelatePoint(contacts[candidateAnchor], normal)).magnitudeSquared();
				if (distance < closestDistance)
				{
					closestDistance = distance;
					closestCluster = candidateCluster;
				}
				candidateAnchors = clearLowestSetBit(candidateAnchors);
				++candidateCluster;
			}
			if (closestCluster == clusterIndex)
				clusterMask |= PxU32(1) << index;
		}
		if (clusterMask)
			selectedMask |= PxU32(1) << selectDcuCorrelateMinSeparation(contacts, clusterMask, anchorMask, clusterBias);
		remainingAnchors = clearLowestSetBit(remainingAnchors);
		++clusterIndex;
	}

	for (PxU32 fillIndex = __popc(anchorMask); fillIndex < SUBMANIFOLD_MAX_CONTACTS; ++fillIndex)
	{
		const PxU32 remainingMask = allMask & ~selectedMask;
		if (!remainingMask)
			break;
		selectedMask |= PxU32(1) << selectDcuCorrelateMinSeparation(contacts, remainingMask, anchorMask, clusterBias);
	}
	return selectedMask;
}

__device__ void correlateDcuSerial(
	const ConvexMeshPair& pair,
	ConvexTriNormalAndIndex* PX_RESTRICT nicBuffer,
	const ConvexTriContacts* PX_RESTRICT contactBuffer,
	PxReal* PX_RESTRICT depthBuffer,
	PxU32* PX_RESTRICT tempCounts,
	PxgPersistentContactMultiManifold* PX_RESTRICT outBuffer,
	const PxReal clusterBias,
	ConvexTriContact* PX_RESTRICT tempConvexTriContacts,
	DcuCorrelateContact* candidates)
{
	const PxU32 start = PxU32(pair.startIndex);
	const PxU32 count = PxU32(pair.count);
	PxU32 nbTriangles = 0;
	for (PxU32 index = 0; index < count; ++index)
	{
		const PxReal depth = depthBuffer[start + index];
		const ConvexTriNormalAndIndex nic = nicBuffer[start + index];
		const PxU32 nbContacts = ConvexTriNormalAndIndex::getNbContacts(nic.index);
		if (depth != FLT_MAX && PxIsFinite(depth) && nbContacts > 0 && isValidDcuCorrelateNormal(nic.normal))
		{
			depthBuffer[start + nbTriangles] = depth;
			tempCounts[start + nbTriangles] = (index << 4) | nbContacts;
			nicBuffer[start + nbTriangles] = nic;
			++nbTriangles;
		}
	}

	PxgPersistentContactMultiManifold& output = outBuffer[pair.cmIndex];
	if (nbTriangles == 0)
	{
		output.mNbManifolds = 0;
		return;
	}

	PxU32 nbPatches = 0;
	PxU32 remainingTriangles = nbTriangles;
	while (nbPatches < MULTIMANIFOLD_MAX_MANIFOLDS && remainingTriangles != 0)
	{
		PxU32 bestIndex = start;
		PxReal bestDepth = PX_MAX_F32;
		PxU32 winningLane = WARP_SIZE;
		for (PxU32 lane = 0; lane < WARP_SIZE; ++lane)
		{
			PxU32 laneBestIndex = start;
			PxReal laneBestDepth = PX_MAX_F32;
			for (PxU32 triangle = lane; triangle < nbTriangles; triangle += WARP_SIZE)
			{
				if (tempCounts[start + triangle] && depthBuffer[start + triangle] < laneBestDepth)
				{
					laneBestDepth = depthBuffer[start + triangle];
					laneBestIndex = start + triangle;
				}
			}
			if (laneBestDepth < bestDepth || (laneBestDepth == bestDepth && lane < winningLane))
			{
				bestDepth = laneBestDepth;
				bestIndex = laneBestIndex;
				winningLane = lane;
			}
		}

		const PxVec3 bestNormal = nicBuffer[bestIndex].normal;
		PxU32 currentContactMask = 0;
		for (PxU32 chunk = 0; chunk < nbTriangles; chunk += WARP_SIZE)
		{
			for (PxU32 lane = 0; lane < WARP_SIZE && chunk + lane < nbTriangles; ++lane)
			{
				const PxU32 compactIndex = start + chunk + lane;
				const PxU32 packed = tempCounts[compactIndex];
				if (!packed)
					continue;

				const ConvexTriNormalAndIndex nic = nicBuffer[compactIndex];
				if (nic.normal.dot(bestNormal) <= PATCH_ACCEPTANCE_EPS)
					continue;

				--remainingTriangles;
				const PxU32 nbContacts = packed & 7;
				const PxU32 originalTriangleOffset = packed >> 4;
				const PxU32 contactStart = contactBuffer[start + originalTriangleOffset].index;
				const PxU32 triangleIndex = ConvexTriNormalAndIndex::getTriangleIndex(nic.index);
				for (PxU32 contact = 0; contact < nbContacts; ++contact)
				{
					const float4 pointSeparation = tempConvexTriContacts[contactStart + contact].contact_sepW;
					if (!isValidDcuCorrelateContact(pointSeparation))
						continue;

					const PxU32 freeMask = ~currentContactMask;
					const PxU32 candidateIndex = lowestSetIndex(freeMask);
					candidates[candidateIndex].pointSeparation = pointSeparation;
					candidates[candidateIndex].triangleIndex = triangleIndex;
					currentContactMask |= PxU32(1) << candidateIndex;
					if (currentContactMask == FULL_MASK)
						currentContactMask = reduceDcuCorrelateContacts(candidates, currentContactMask, bestNormal, clusterBias);
				}
				tempCounts[compactIndex] = 0;
			}
			if (__popc(currentContactMask) > SUBMANIFOLD_MAX_CONTACTS)
				currentContactMask = reduceDcuCorrelateContacts(candidates, currentContactMask, bestNormal, clusterBias);
		}

		const PxU32 nbContacts = __popc(currentContactMask);
		if (nbContacts)
		{
			const PxTransform aToB = pair.aToB;
			const PxVec3 normalInB = aToB.rotate(bestNormal);
			if (!isValidDcuCorrelateNormal(normalInB))
				continue;

			PxU32 outputContact = 0;
			for (PxU32 scanMask = currentContactMask; scanMask; scanMask = clearLowestSetBit(scanMask))
			{
				const PxU32 candidateIndex = lowestSetIndex(scanMask);
				const float4 pointSeparation = candidates[candidateIndex].pointSeparation;
				const PxVec3 pointA(pointSeparation.x, pointSeparation.y, pointSeparation.z);
				const PxVec3 pointB = aToB.transform(pointA) + normalInB * pointSeparation.w;
				if (!pointB.isFinite())
					continue;

				PxgContact& contact = output.mContacts[nbPatches][outputContact++];
				contact.normal = -normalInB;
				contact.pointA = pointA;
				contact.pointB = pointB;
				contact.penetration = pointSeparation.w;
				contact.triIndex = candidates[candidateIndex].triangleIndex;
			}
			if (outputContact)
			{
				output.mNbContacts[nbPatches] = outputContact;
				++nbPatches;
			}
		}
	}

	if (nbPatches)
	{
		output.mRelativeTransform.q = PxAlignedQuat(pair.aToB.q);
		output.mRelativeTransform.p = make_float4(pair.aToB.p.x, pair.aToB.p.y, pair.aToB.p.z, 0.f);
	}
	output.mNbManifolds = nbPatches;
}
#endif

__device__
void correlate(
	const ConvexMeshPair* PX_RESTRICT				meshPairBuffer,
	ConvexTriNormalAndIndex ** PX_RESTRICT			nicBufferPtr,
	const ConvexTriContacts ** PX_RESTRICT			contactBufferPtr,
	PxReal ** PX_RESTRICT							depthBufferPtr,
	PxU32* PX_RESTRICT								tempCounts,
	Scratch* PX_RESTRICT							s,
	PxgPersistentContactMultiManifold* PX_RESTRICT	outBuffer,
	PxsContactManagerOutput* PX_RESTRICT			cmOutputs,
	const PxU32										numPairs,
	const PxReal									clusterBias,
	ConvexTriContact* PX_RESTRICT					tempConvexTriContacts,
	PxU32* PX_RESTRICT								pTempContactIndex
)
{
	__shared__ PxU32 triangleRunSums[CORRELATE_WARPS_PER_BLOCK][WARP_SIZE];
	__shared__ PxU32 startIndices[CORRELATE_WARPS_PER_BLOCK][WARP_SIZE];
	__shared__ PxU32 triIndices[CORRELATE_WARPS_PER_BLOCK][WARP_SIZE];
#if defined(PX_DCU_PORT) && PX_DCU_PORT
	__shared__ DcuCorrelateContact dcuCandidates[CORRELATE_WARPS_PER_BLOCK][WARP_SIZE];
#endif

	PxU32* triRunSums = triangleRunSums[threadIdx.y];
	PxU32* triStartIndex = startIndices[threadIdx.y];
	PxU32* triInds = triIndices[threadIdx.y];

	const int pairIndex = threadIdx.y + CORRELATE_WARPS_PER_BLOCK * blockIdx.x;

	if (pairIndex >= numPairs)
		return;

#if defined(PX_DCU_PORT) && PX_DCU_PORT
	const int dcuCount = meshPairBuffer[pairIndex].count;
	if (dcuCount == CONVEX_TRIMESH_CACHED)
		return;
	if (dcuCount >= 0 && dcuCount <= WARP_SIZE * 4)
	{
		if (threadIdx.x == 0)
		{
			const ConvexMeshPair pair = meshPairBuffer[pairIndex];
			correlateDcuSerial(
				pair, *nicBufferPtr, *contactBufferPtr, *depthBufferPtr,
				tempCounts, outBuffer, clusterBias, tempConvexTriContacts, dcuCandidates[threadIdx.y]);
		}
		return;
	}
#endif

	s->pairIndex = pairIndex;
	const int tI = threadIdx.x;
	if (tI < sizeof(ConvexMeshPair) / 4)
		((int*)(&s->pair))[tI] = ((int*)(meshPairBuffer + pairIndex))[tI];

	__syncwarp();

	int start = s->pair.startIndex, count = s->pair.count;

	if (count == CONVEX_TRIMESH_CACHED)
	{
		// We've hit cache, nothing to do here
		return;
	}

	ConvexTriNormalAndIndex* PX_RESTRICT nicBuffer = *nicBufferPtr;
	const ConvexTriContacts* PX_RESTRICT contactBuffer = *contactBufferPtr;
	PxReal* PX_RESTRICT depthBuffer = *depthBufferPtr;


	PxU32 nbTriangles = purgeEmptyTriangles(FULL_MASK, depthBuffer + start, nicBuffer + start, count, tempCounts + start);

	__syncwarp();


	if (nbTriangles == 0)
	{
		outBuffer[pairIndex].mNbManifolds = 0;
		return;
	}

	//Now, we loop creating patches and contacts from the triangles that actually have contacts!

	PxU32 nbPatches = 0;
	PxU32 numContactsTotal = 0;

	PxU32 remainingTriangles = nbTriangles;



	for (; nbPatches < MULTIMANIFOLD_MAX_MANIFOLDS && remainingTriangles != 0;)
	{
		//We loop through all triangles, load contacts, compress etc...
		//(1) Find deepest triangle...


		int bestIndex = 0;
		PxReal bestDepth = PX_MAX_F32;
		for (PxU32 i = threadIdx.x; i < nbTriangles; i += 32)
		{
			if (i < nbTriangles)
			{
				if (tempCounts[start + i]) //Still has contacts!
				{
					PxReal d = depthBuffer[start + i];
					if (d < bestDepth)
					{
						bestDepth = d;
						bestIndex = start + i;
					}
				}
			}
		}

		int i0 = minIndex(bestDepth, FULL_MASK, bestDepth);

		bestIndex = __shfl_sync(FULL_MASK, bestIndex, i0);

		//Now we know which triangle defines this plane...

		PxVec3 bestNormal = nicBuffer[bestIndex].normal;

		PxU32 currentContactMask = 0;


		PxVec3 pA(0.f);
		PxReal separation = 0.f;
		PxU32 contactTriangleIndex = 0;
		bool hasContact = false;

		for (PxU32 i = 0; i < nbTriangles; i += 32)
		{
			PxVec3 normal(0.f);
			PxU32 triIndex = 0;
			PxU32 nbContacts = 0;
			PxU32 startIndex = 0;

			if ((i + threadIdx.x) < nbTriangles)
			{
				PxU32 index = tempCounts[start + i + threadIdx.x];
				//We read contacts from the temp contacts buffer because this will get cleared when the contacts are considered
				//for a patch...
				nbContacts = index & 7;
				startIndex = index >> 4;
				if (nbContacts)
				{
					ConvexTriNormalAndIndex nic = nicBuffer[start + i + threadIdx.x];
					normal = nic.normal;
					triIndex = ConvexTriNormalAndIndex::getTriangleIndex(nic.index);
				}
			}

			bool accept = normal.dot(bestNormal) > PATCH_ACCEPTANCE_EPS;

			if(!accept)
				nbContacts = 0;


			PxU32 acceptMask = __ballot_sync(FULL_MASK, accept);

			remainingTriangles -= __popc(acceptMask);

			PxU32 summedContacts = warpScan<AddOpPxU32, PxU32>(FULL_MASK, nbContacts);

			triRunSums[threadIdx.x] = summedContacts - nbContacts;
			triStartIndex[threadIdx.x] = startIndex;
			triInds[threadIdx.x] = triIndex;

			PxU32 totalContacts = __shfl_sync(FULL_MASK, summedContacts, 31);
			__syncwarp();

			for (PxU32 c = 0; c < totalContacts;)
			{
				//calculate my read index (where I'm offset from...)
				const PxU32 readMask = ~currentContactMask;
				bool needsContact = readMask & (PxU32(1) << threadIdx.x);
				PxU32 readIndex = c + warpScanExclusive(readMask, threadIdx.x);

				if (needsContact && readIndex < totalContacts)
				{
					PxU32 idx = binarySearch<PxU32>(triRunSums, 32u, readIndex);

					PxU32 offset = readIndex - triRunSums[idx];

					PxU32 startIdx = triStartIndex[idx] + start;

					const PxU32 contactStartIdx = contactBuffer[startIdx].index;


					float4 v = tempConvexTriContacts[contactStartIdx + offset].contact_sepW;

					pA = PxVec3(v.x, v.y, v.z);
					separation = v.w;
					contactTriangleIndex = triInds[idx];
					hasContact = true;
				}

				currentContactMask = __ballot_sync(FULL_MASK, hasContact); //Mask every thread that has a contact!

				c += __popc(readMask);
				PxU32 nbContacts = __popc(currentContactMask);

				if (nbContacts > SUBMANIFOLD_MAX_CONTACTS)
				{
					currentContactMask = contactReduce<true, false, SUBMANIFOLD_MAX_CONTACTS, false>(pA, separation, bestNormal, currentContactMask, clusterBias);
				}
				hasContact = currentContactMask & (PxU32(1) << threadIdx.x);
			}

			__syncwarp(); //triRunSums is read and written in the same loop - separate read and write with syncs

			if (accept)
			{
				tempCounts[start + i + threadIdx.x] = 0; //Clear this triangle - it is done!
			}

		}

		__syncwarp();
		PxU32 nbContacts = __popc(currentContactMask);

		if (currentContactMask != 0)
		{
			PxgContact * cp = outBuffer[s->pair.cmIndex].mContacts[nbPatches] + warpScanExclusive(currentContactMask, tI);

			if (threadIdx.x == 0)
				outBuffer[s->pair.cmIndex].mNbContacts[nbPatches] = nbContacts;

			numContactsTotal += nbContacts;
			if (currentContactMask & (PxU32(1) << tI))
			{
				PxVec3 nor = ldS(s->pair.aToB).rotate(bestNormal);

				cp->normal = -nor;
				cp->pointA = pA;
				// TODO: PxTransform::transform is incredibly profligate with registers (i.e. the write-out phase uses
				// more regs than the contact culling phase). Better would be to stash the transform in matrix form
				// when not under register pressure (i.e. when initially loading it) then use a custom shmem
				//  matrix multiply, which requires essentially no tmps

				PxVec3 pointB = ldS(s->pair.aToB).transform(pA) + nor * separation;

				cp->pointB = pointB;
				cp->penetration = separation;
				cp->triIndex = contactTriangleIndex;

				/*printf("cp normal(%f, %f, %f)\n", cp->normal.x, cp->normal.y, cp->normal.z);
				printf("cp pointA(%f, %f, %f)\n", pA.x, pA.y, pA.z);
				printf("cp pointB(%f, %f, %f)\n", pointB.x, pointB.y, pointB.z);*/
			}

			nbPatches++;
		}
	}

	if (nbPatches > 0)
	{
		// TODO avoroshilov: decide what to do with aligned transforms
		PxAlignedTransform aToB_aligned;// = s->pair.aToB;
		aToB_aligned.p.x = s->pair.aToB.p.x;
		aToB_aligned.p.y = s->pair.aToB.p.y;
		aToB_aligned.p.z = s->pair.aToB.p.z;

		aToB_aligned.q.q.x = s->pair.aToB.q.x;
		aToB_aligned.q.q.y = s->pair.aToB.q.y;
		aToB_aligned.q.q.z = s->pair.aToB.q.z;
		aToB_aligned.q.q.w = s->pair.aToB.q.w;

		PxAlignedTransform_WriteWarp(&outBuffer[s->pair.cmIndex].mRelativeTransform, aToB_aligned);
	}

	outBuffer[s->pair.cmIndex].mNbManifolds = nbPatches;
}


extern "C" __global__
void convexTrimeshCorrelate(
	const ConvexMeshPair * PX_RESTRICT				pairs,
	PxReal ** PX_RESTRICT							depthsPtr,
	ConvexTriNormalAndIndex ** PX_RESTRICT			normalAndIndexPtr,
	const ConvexTriContacts ** PX_RESTRICT			contactsPtr,

	PxgPersistentContactMultiManifold * PX_RESTRICT	output,
	PxsContactManagerOutput * PX_RESTRICT			cmOutputs,
	PxU32* PX_RESTRICT								tempCountAndOffsetBuffer,
	const PxU32										numPairs,
	const PxReal									clusterBias,
	ConvexTriContact* PX_RESTRICT					tempConvexTriContacts,
	PxU32*											pTempContactIndex
)
{
	const int warpIndex = threadIdx.y;
	__shared__ char scratch[sizeof(Scratch) *  CORRELATE_WARPS_PER_BLOCK];
#if defined(PX_DCU_PORT) && PX_DCU_PORT
	if (numPairs == 0)
	{
		if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
		{
			const volatile char* version = gPxgDcuCorrelateStageVersion;
			volatile char* markerScratch = scratch;
			markerScratch[0] = version[0];
		}
		return;
	}
#endif
	correlate(
		pairs,
		normalAndIndexPtr,
		contactsPtr,
		depthsPtr,
		tempCountAndOffsetBuffer,
		((Scratch*)&scratch) + warpIndex,
		output,
		cmOutputs,
		numPairs,
		clusterBias,
		tempConvexTriContacts,
		pTempContactIndex
	);
}
