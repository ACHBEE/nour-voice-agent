"""Exit 0 if the RAG service answers the standard gRPC health check as SERVING (Docker probe).

python scripts/grpc_health_check.py [host:port]
"""

import sys

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


def main() -> int:
    addr = sys.argv[1] if len(sys.argv) > 1 else "localhost:50051"
    with grpc.insecure_channel(addr) as channel:
        try:
            resp = health_pb2_grpc.HealthStub(channel).Check(
                health_pb2.HealthCheckRequest(service="nour.rag.v1.RagService"), timeout=2
            )
        except grpc.RpcError:
            return 1
    return 0 if resp.status == health_pb2.HealthCheckResponse.SERVING else 1


if __name__ == "__main__":
    sys.exit(main())
