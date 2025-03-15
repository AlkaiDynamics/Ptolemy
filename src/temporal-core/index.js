const fs = require('fs').promises;
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const { logger } = require('../../config/config');

class TemporalCore {
  constructor(storagePath) {
    this.storagePath = storagePath;
    this.currentEvents = [];
  }

  async initialize() {
    try {
      await fs.mkdir(this.storagePath, { recursive: true });
      logger.info(`Temporal Core initialized with storage path: ${this.storagePath}`);
    } catch (error) {
      logger.error(`Failed to initialize Temporal Core: ${error.message}`);
      throw error;
    }
  }

  async recordEvent(eventType, eventData) {
    const event = {
      id: uuidv4(),
      timestamp: new Date().toISOString(),
      type: eventType,
      data: eventData
    };

    try {
      const eventFilePath = path.join(this.storagePath, `${event.id}.json`);
      await fs.writeFile(eventFilePath, JSON.stringify(event, null, 2));
      this.currentEvents.push(event);
      logger.info(`Event recorded: ${eventType} (${event.id})`);
      return event;
    } catch (error) {
      logger.error(`Failed to record event: ${error.message}`);
      throw error;
    }
  }

  async getEvents(filters = {}) {
    try {
      const files = await fs.readdir(this.storagePath);
      const events = [];

      for (const file of files) {
        if (path.extname(file) === '.json') {
          const filePath = path.join(this.storagePath, file);
          const eventData = JSON.parse(await fs.readFile(filePath, 'utf8'));
          
          // Apply filters if any
          let includeEvent = true;
          for (const [key, value] of Object.entries(filters)) {
            if (key === 'type' && eventData.type !== value) {
              includeEvent = false;
              break;
            }
            if (key === 'timeAfter' && new Date(eventData.timestamp) <= new Date(value)) {
              includeEvent = false;
              break;
            }
            if (key === 'timeBefore' && new Date(eventData.timestamp) >= new Date(value)) {
              includeEvent = false;
              break;
            }
          }

          if (includeEvent) {
            events.push(eventData);
          }
        }
      }

      return events.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    } catch (error) {
      logger.error(`Failed to get events: ${error.message}`);
      throw error;
    }
  }

  async getEventById(eventId) {
    try {
      const eventFilePath = path.join(this.storagePath, `${eventId}.json`);
      const eventData = JSON.parse(await fs.readFile(eventFilePath, 'utf8'));
      return eventData;
    } catch (error) {
      logger.error(`Failed to get event by ID: ${error.message}`);
      throw error;
    }
  }

  async getEventStream(startTime = null, endTime = null) {
    const filters = {};
    if (startTime) filters.timeAfter = startTime;
    if (endTime) filters.timeBefore = endTime;
    return this.getEvents(filters);
  }
}

module.exports = TemporalCore;
