import os
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Tuple
import boto3
import uuid
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.context import Context, attach, detach

# Global tracer instance - initialized only once when the module is imported
_TRACER = None

def get_tracer():
    """Singleton pattern to get or initialize the tracer"""
    global _TRACER
    if _TRACER is None:
        # Initialize tracer provider
        tracer_provider = TracerProvider()
        
        # Configure the OTLP exporter
        otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        
        # Add batch processor to the tracer provider
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        
        # Set the tracer provider
        trace.set_tracer_provider(tracer_provider)
        
        # Instrument requests library
        RequestsInstrumentor().instrument()
        
        # Create the tracer
        _TRACER = trace.get_tracer(__name__)
    
    return _TRACER

def create_request_context() -> Context:
    """Create a new context for each request with unique trace ID"""
    propagator = TraceContextTextMapPropagator()
    
    # Create a carrier with unique trace information
    carrier = {
        "traceparent": f"00-{uuid.uuid4().hex}-{uuid.uuid4().hex[:16]}-01"
    }
    
    # Extract context from carrier
    return propagator.extract(carrier=carrier)

def model_fn(model_dir: str) -> Any:
    """Load the model from disk."""
    tracer = get_tracer()
    with tracer.start_as_current_span("model_fn") as span:
        try:
            span.set_attribute("model_dir", model_dir)
            
            # Your model loading logic here
            # Example:
            # model = joblib.load(os.path.join(model_dir, "model.joblib"))
            model = "placeholder_model"  # Replace with your actual model loading
            
            span.set_attribute("model_loaded", True)
            return model
            
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            span.record_exception(e)
            raise

def transform_fn(model: Any, request_body: bytes, content_type: str) -> Tuple[bytes, str]:
    """Transform the input data into predictions."""
    # Get the singleton tracer instance
    tracer = get_tracer()
    
    # Create a new context for this request
    request_context = create_request_context()
    
    # Attach the context and get a token
    token = attach(request_context)
    
    try:
        # Start the main span for this transform call
        with tracer.start_as_current_span("transform_fn") as span:
            # Add request ID to trace
            request_id = str(uuid.uuid4())
            span.set_attribute("request_id", request_id)
            span.set_attribute("content_type", content_type)
            span.set_attribute("request_size", len(request_body))
            
            # Start nested span for input processing
            with tracer.start_span("process_input") as input_span:
                if content_type == "application/json":
                    input_data = json.loads(request_body.decode())
                    input_span.set_attribute("input_format", "json")
                    input_span.set_attribute("input_keys", str(list(input_data.keys())))
                else:
                    raise ValueError(f"Unsupported content type: {content_type}")

            # Start nested span for model inference
            with tracer.start_span("model_inference") as inference_span:
                # Your prediction logic here
                # Example:
                # predictions = model.predict(input_data)
                predictions = {"result": "placeholder_prediction"}  # Replace with actual prediction
                inference_span.set_attribute("prediction_generated", True)

            # Start nested span for output processing
            with tracer.start_span("process_output") as output_span:
                response_body = json.dumps(predictions).encode('utf-8')
                output_span.set_attribute("response_size", len(response_body))

            return response_body, "application/json"

    except Exception as e:
        if span:  # Check if span exists
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            span.record_exception(e)
        traceback.print_exc()
        raise
        
    finally:
        # Always detach the context
        detach(token)

def error_handler(error: Exception, context: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Handle errors during inference with tracing."""
    tracer = get_tracer()
    with tracer.start_as_current_span("error_handler") as span:
        span.set_attribute("error", True)
        span.set_attribute("error.type", type(error).__name__)
        span.set_attribute("error.message", str(error))
        span.record_exception(error)
        
        error_response = {
            "error": str(error),
            "type": type(error).__name__,
            "timestamp": datetime.utcnow().isoformat()
        }
        return json.dumps(error_response).encode('utf-8'), "application/json"
