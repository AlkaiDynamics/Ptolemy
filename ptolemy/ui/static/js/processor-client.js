/**
 * Processor Client Library
 * 
 * A client-side JavaScript library for interacting with the Multi-Model Processor API.
 * This module provides a clean interface for processing tasks, streaming responses,
 * and monitoring cache performance.
 */

class ProcessorClient {
    /**
     * Initialize a new processor client.
     * @param {Object} options - Configuration options
     * @param {string} options.baseUrl - Base URL for the processor API
     * @param {Function} options.onError - Error handler callback
     */
    constructor(options = {}) {
        this.baseUrl = options.baseUrl || '/processor';
        this.onError = options.onError || console.error;
        this.specs = null;
        this.models = [];
    }

    /**
     * Initialize the client by fetching API specifications.
     * @returns {Promise<Object>} API specifications
     */
    async initialize() {
        try {
            const response = await fetch(`${this.baseUrl}/api-specs`);
            
            if (!response.ok) {
                throw new Error(`Failed to initialize: ${response.status} ${response.statusText}`);
            }
            
            this.specs = await response.json();
            this.models = this.specs.models || [];
            return this.specs;
        } catch (error) {
            this.onError('Initialization error', error);
            throw error;
        }
    }

    /**
     * Get available models.
     * @returns {Promise<Array>} List of available models
     */
    async getModels() {
        try {
            const response = await fetch(`${this.baseUrl}/models`);
            
            if (!response.ok) {
                throw new Error(`Failed to get models: ${response.status} ${response.statusText}`);
            }
            
            const data = await response.json();
            this.models = data.models || [];
            return this.models;
        } catch (error) {
            this.onError('Error fetching models', error);
            throw error;
        }
    }

    /**
     * Process a task.
     * @param {Object} options - Processing options
     * @param {string} options.task - The task to process
     * @param {string} [options.modelPreference] - Preferred model
     * @param {Object} [options.parameters] - Model parameters
     * @param {boolean} [options.bypassCache] - Whether to bypass cache
     * @returns {Promise<Object>} Processing result
     */
    async processTask(options) {
        try {
            const { task, modelPreference, parameters, bypassCache } = options;
            
            if (!task) {
                throw new Error('Task is required');
            }
            
            const payload = {
                task,
                model_preference: modelPreference,
                parameters,
                bypass_cache: bypassCache
            };
            
            const response = await fetch(`${this.baseUrl}/process`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server error: ${response.status} - ${errorText}`);
            }
            
            return await response.json();
        } catch (error) {
            this.onError('Processing error', error);
            throw error;
        }
    }

    /**
     * Process a task with streaming response.
     * @param {Object} options - Processing options
     * @param {string} options.task - The task to process
     * @param {string} [options.modelPreference] - Preferred model
     * @param {Object} [options.parameters] - Model parameters
     * @param {Function} options.onChunk - Callback for each chunk
     * @param {Function} [options.onComplete] - Callback when stream completes
     * @param {Function} [options.onError] - Callback for stream errors
     * @returns {Promise<void>}
     */
    async streamTask(options) {
        const { task, modelPreference, parameters, onChunk, onComplete, onError } = options;
        
        if (!task) {
            throw new Error('Task is required');
        }
        
        if (!onChunk || typeof onChunk !== 'function') {
            throw new Error('onChunk callback is required');
        }
        
        const errorHandler = onError || this.onError;
        
        try {
            const payload = {
                task,
                model_preference: modelPreference,
                parameters
            };
            
            const response = await fetch(`${this.baseUrl}/process-stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server error: ${response.status} - ${errorText}`);
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            
            try {
                while (true) {
                    const { done, value } = await reader.read();
                    
                    if (done) {
                        // Process any remaining data in buffer
                        if (buffer.trim()) {
                            this._processEventData(buffer, onChunk, errorHandler);
                        }
                        break;
                    }
                    
                    // Decode chunk and add to buffer
                    const chunk = decoder.decode(value, { stream: true });
                    buffer += chunk;
                    
                    // Process complete SSE events
                    const events = buffer.split('\n\n');
                    buffer = events.pop() || ''; // Keep the last incomplete event in buffer
                    
                    for (const event of events) {
                        if (event.trim()) {
                            this._processEventData(event, onChunk, errorHandler);
                        }
                    }
                }
                
                if (onComplete && typeof onComplete === 'function') {
                    onComplete();
                }
            } catch (error) {
                errorHandler('Stream processing error', error);
                throw error;
            }
        } catch (error) {
            errorHandler('Stream initialization error', error);
            throw error;
        }
    }

    /**
     * Get cache statistics.
     * @returns {Promise<Object>} Cache statistics
     */
    async getCacheStats() {
        try {
            const response = await fetch(`${this.baseUrl}/cache-stats`);
            
            if (!response.ok) {
                throw new Error(`Failed to get cache stats: ${response.status} ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            this.onError('Error fetching cache stats', error);
            throw error;
        }
    }

    /**
     * Clear the cache.
     * @returns {Promise<Object>} Clear cache result
     */
    async clearCache() {
        try {
            const response = await fetch(`${this.baseUrl}/clear-cache`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error(`Failed to clear cache: ${response.status} ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            this.onError('Error clearing cache', error);
            throw error;
        }
    }

    /**
     * Get error analysis data.
     * @returns {Promise<Object>} Error analysis data
     */
    async getErrorAnalysis() {
        try {
            const response = await fetch(`${this.baseUrl}/error-analysis`);
            
            if (!response.ok) {
                throw new Error(`Failed to get error analysis: ${response.status} ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            this.onError('Error fetching error analysis', error);
            throw error;
        }
    }

    /**
     * Process SSE event data.
     * @private
     * @param {string} eventText - Event text
     * @param {Function} onChunk - Chunk callback
     * @param {Function} onError - Error callback
     */
    _processEventData(eventText, onChunk, onError) {
        if (!eventText.trim().startsWith('data:')) {
            return;
        }
        
        try {
            const dataText = eventText.trim().substring(5).trim();
            const data = JSON.parse(dataText);
            onChunk(data);
        } catch (error) {
            onError('Error parsing event data', error, eventText);
        }
    }
}

// Export the client class
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ProcessorClient };
} else {
    window.ProcessorClient = ProcessorClient;
}
