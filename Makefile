BUILD_DIR=build

all:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR); cmake .. && $(MAKE)

clean:
	@rm -rf $(BUILD_DIR)
