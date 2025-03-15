require('dotenv').config();
const winston = require('winston');

const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(winston.format.timestamp(), winston.format.json()),
  transports: [
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' }),
    new winston.transports.Console({ format: winston.format.simple() })
  ]
});

const config = {
  openai: {
    apiKey: process.env.OPENAI_API_KEY,
    defaultModel: process.env.DEFAULT_MODEL || 'gpt-4',
  },
  temporalCore: {
    storagePath: process.env.TEMPORAL_STORAGE_PATH || './data/temporal',
  },
  contextEngine: {
    storagePath: process.env.CONTEXT_STORAGE_PATH || './data/context',
  }
};

module.exports = { config, logger };
