# Shared Service Infrastructure

Common utilities and patterns for all platform services.

## Port Management

All services use a centralized port registry to ensure unique ports:

```python
from services.shared.server import SERVICE_PORTS, get_service_port

# Ports are pre-assigned:
# - sessions: 50052
# - models: 50053
# - data: 50054
# - guardrails: 50055
# - tools: 50056
# - workflow: 50058
# - observability: 50059
# - experiments: 50060

# Get port for a service (checks env var first, then registry)
port = get_service_port("sessions")  # Returns 50052 or SESSIONS_PORT if set
```

Ports can be overridden via environment variables:
- `SESSIONS_PORT=50060` (overrides default 50052)
- `MODELS_PORT=50061` (overrides default 50053)
- etc.

## Creating a Service

All services follow the same pattern:

### 1. Create Store (Data Layer)

```python
# services/my_service/store.py
class MyServiceStore:
    """Data persistence layer."""
    def __init__(self):
        self._data = {}
    
    def get_data(self, key):
        return self._data.get(key)
```

### 2. Create Service (Business Logic)

```python
# services/my_service/service.py
from proto import my_service_pb2_grpc
from services.shared.servicer_base import BaseServicer
from .store import MyServiceStore

class MyService(my_service_pb2_grpc.MyServiceServicer, BaseServicer):
    def __init__(self):
        self.store = MyServiceStore()
    
    def add_to_server(self, server):
        """Required: Add servicer to gRPC server."""
        my_service_pb2_grpc.add_MyServiceServicer_to_server(self, server)
    
    def MyMethod(self, request, context):
        """Implement gRPC methods."""
        return self.store.get_data(request.key)
```

### 3. Create Main Entry Point

```python
# services/my_service/main.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.shared.server import create_grpc_server, run_service, get_service_port
from .service import MyService

def main():
    service_name = "my_service"
    port = get_service_port(service_name)
    
    servicer = MyService()
    server = create_grpc_server(
        servicer=servicer,
        port=port,
        service_name=service_name
    )
    
    run_service(server, service_name, port)

if __name__ == "__main__":
    main()
```

## Benefits

1. **Consistent Structure**: All services follow the same pattern
2. **Unique Ports**: Centralized registry prevents conflicts
3. **Shared Utilities**: Common server setup code reused
4. **Separation of Concerns**: Store (data) vs Service (logic) vs Main (entry point)
5. **Easy to Extend**: Add new services by following the pattern
