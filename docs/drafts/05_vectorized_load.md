# Step 5. Coalescing + Vectorized Load (한글 초안)

> 최종 영어 문서 작성용 참고 초안. 수치는 최신 실측(00: 253ms 기준 런)을 사용했으며,
> 최종본에는 README에 채택할 런의 수치로 교체할 것.

## 1. 개념

### 복습: step 04가 남긴 문제

step 04에서 attention 전체를 커널 하나로 합쳤지만(fusion) baseline보다도 느려졌다
(341ms, 0.74x). 커널을 "합치는 것"과 합쳐진 커널의 "메모리 접근이 효율적인 것"은
별개의 문제다. step 05~07은 fused 커널의 메모리 접근을 한 층씩 고치는 과정이고,
그 첫 번째가 전송 단위다.

### Coalescing(병합 접근) — 워프 단위로 생각하기

GPU가 HBM(글로벌 메모리)에 접근하는 실질 단위는 스레드 1개가 아니라 **워프(32스레드)**다.
한 워프의 32개 로드가 **연속된 주소**를 가리키면, 하드웨어는 이를 소수의 큰 메모리
트랜잭션으로 합쳐 처리한다. 이것이 coalescing이다. 주소가 흩어져 있으면 같은 데이터를
가져오는 데 훨씬 많은 트랜잭션이 쪼개져 나가고 대역폭이 낭비된다.

step 04의 협동 로드 루프는

```cuda
for (int idx = threadIdx.x; idx < BC * d; idx += blockDim.x)
```

연속 스레드가 연속 주소를 읽는 구조라 **이미 coalesced 상태였다**.
step 05가 새로 더한 것은 "벡터화"다.

### Vectorized Load — 4개씩 나르기

float 하나(4B)씩 나르던 것을 `float4`(16B) 단위로 나른다. 효과는 두 가지다.

1. **로드 명령 수가 1/4**: `LDG.32` 4개가 `LDG.128` 1개로. 명령 발행과 주소 계산
   오버헤드가 그만큼 사라진다.
2. **워프당 요청량 128B → 512B**: 같은 메모리 지연시간 동안 더 많은 바이트가
   "비행 중(in flight)"이 되어 대역폭 활용이 올라간다.

```
스칼라:  lane0 lane1 lane2 ... lane31   →  32 x 4B  = 128B / 명령
float4:  lane0 [----] lane1 [----] ...  →  32 x 16B = 512B / 명령
```

### 왜 2.9배나 빨라졌나 — 진짜 이유는 shared memory 쪽

글로벌 로드 개선만으로 2.9배(341→119ms)는 설명되지 않는다. 핵심은 QK^T 내적 루프의
**shared memory 읽기도 float4가 됐다**는 점이다.

step 04의 `Ktile[lane * d + k]`는 32개 lane이 전부 같은 뱅크를 때리는 32-way 뱅크
충돌이라 읽기가 32배로 직렬화되고 있었다(자세한 구조는 step 06에서). float4로 읽으면
"충돌이 나는 접근"의 횟수 자체가 1/4로 줄어 총 직렬화 사이클이 크게 감소한다.
**충돌 자체는 아직 남아 있다** — 그것을 없애는 것이 step 06이다.

### 제약

float4로 읽으려면 각 행이 4개 단위로 나눠 떨어지고 16B 정렬이 맞아야 한다
→ `d % 4 == 0` 검사 추가. (torch 텐서의 시작 주소는 충분히 정렬돼 있음)

## 2. 코드 구현 설명

### 재해석 캐스트와 인덱스 단위 변경

```cuda
const int d4 = d / 4;  // 행 길이를 float4 단위로
const float4* Kb4 = reinterpret_cast<const float4*>(Kb);
float4* Ktile4 = reinterpret_cast<float4*>(Ktile);
```

같은 메모리를 float4 배열로 재해석한다. 이후 모든 인덱스 산술이 `idx / d4`,
`idx % d4`처럼 float4 단위로 바뀐다. shared memory 선언에는 `__align__(16)`을 붙여
시작 주소 정렬을 보장한다.

### 로드 루프 (before / after)

```cuda
// step 04: float 1개씩
Ktile[idx] = in ? Kb[(size_t)r * d + c] : 0.0f;

// step 05: float4 1개씩 (경계 밖 행은 zero4로 패딩)
Ktile4[idx] = in ? Kb4[(size_t)r * d4 + c] : zero4;
```

### 내적 루프

```cuda
const float4* q4 = reinterpret_cast<const float4*>(Qtile + warp * d);
const float4* k4 = reinterpret_cast<const float4*>(Ktile + lane * d);
for (int k = 0; k < d4; ++k) {
    const float4 a = q4[k];
    const float4 b = k4[k];
    dot += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
```

로드 1번에 곱-합 4개. 반복 횟수도 d → d/4.

### 바꾸지 않은 곳과 그 이유

- **V 읽기(PV 단계)**: `Vtile[c * d + lane + 32*i]` — lane 하나가 잡는 원소들이
  연속이 아니라(32칸 간격) float4로 묶을 수 없다. 대신 워프 전체로 보면 연속
  워드를 읽으므로 이미 conflict-free/coalesced다.
- **O 쓰기**: 같은 이유로 스칼라 유지. 워프 관점에서는 coalesced.

### 호스트 코드

`attention.cu`에 `TORCH_CHECK(d % 4 == 0, ...)` 추가. 나머지는 step 04와 동일.

## 3. 벤치마크 참고

- **필수**: 메인 표의 04→05 speedup (~2.9x).
- **선택(주장 뒷받침용)**: ncu로 04 vs 05의 글로벌 로드 명령 수
  (`smsp__inst_executed_op_global_ld`)와 shared memory 대기 stall
  (`short_scoreboard`) 비교. "명령 수 1/4" 주장을 실측으로 보여줄 수 있다.
